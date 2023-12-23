import os
os.environ['MKL_NUM_THREADS'] = "1"

from joblib import Memory
from pypesto.sample.auto_correlation import autocorrelation_sokal
from itertools import starmap
import pickle
from AlonsoMarderModel import AlonsoMarderModel
import pypesto
import numpy as np
import pypesto.sample as sample
from pypesto.objective.priors import get_parameter_prior_dict
import pandas as pd
from typing import Dict, Union
import datetime
import h5py
from multiprocess import Manager, Process, Pool, get_context
import multiprocessing as mp
import warnings
import copy
import logging
logger = logging.getLogger(__name__)


MEM_CACHE = Memory('./cache/')


def generated_fn(sample_params: np.ndarray, step: float, stop: float, current_injected: float = 0.0,
                 candidate_model_output: Union[dict, None] = None,
                 del_ty: bool = True) -> Dict[str, Union[bool, np.ndarray, float]]:
    """
    calculates estimated spike times using estimated params

    :param sample_params: estimated parameters
    :param step: time stepping
    :param stop: time stopping
    :param current_injected: time steps
    :param del_ty:
    :param candidate_model_output:
    :return: estimated model
    """
    conductances = {
        'g_Na': sample_params[0],  # 1.0764e3,  # uS, transient sodium conductance
        'g_CaT': sample_params[1],  # 6.4056e0,  # uS, low-threshold calcium conductance
        'g_CaS': sample_params[2],  # 1.0048e1,  # uS, slow calcium conductance
        'g_A': sample_params[3],  # 8.0384e0,  # uS, transient potassium conductance
        'g_KCa': sample_params[4],  # 1.7584e1,  # uS, calcium-dependent potassium conductance
        'g_Kd': sample_params[5],  # 1.240928e2,  # uS, potassium conductance
        'g_H': sample_params[6],  # 1.1304e-1,  # uS, hyperpolarization-activated cation conductance
        'g_L': sample_params[7],  # 1.7584e-1,  # uS, leak conductance
    }
    time_steps = np.arange(0.0, stop, step)
    the_model = AlonsoMarderModel(injected_current=current_injected,
                                  conductances=conductances,
                                  tau_ca=sample_params[8])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_out = the_model.run_simulation(time_steps)
            model_out['success'] = True
    except ValueError as e:
        model_out = {'success': False}
        return model_out
    if del_ty:
        del model_out['t']
        del model_out['y']
    if candidate_model_output is not None:
        candidate_model_output[current_injected] = model_out
        logging.debug(f'generated_fn {current_injected}: {conductances} complete...')
    else:
        logging.debug(f'generated_fn {current_injected}: {conductances} complete...')
        return model_out


def calc_alonso_marder_loss(true_data: dict,  # dict of currents, then
                            candidate_data: dict,
                            alpha=1000.,
                            beta=1000.,
                            gamma=0.,  # gamma=100.,
                            delta=0.,  # delta=100.,
                            nu=0.) -> float:
    """
    Calculate Alonso Marder loss using their loss function

    @param true_data: true data
    @param candidate_data: estimated data
    @param alpha: coefficient for error burst frequency
    @param beta: coefficient for error duty cycle
    @param gamma: coefficient for error mid crossings
    @param delta: coefficient for error slow wave crossings
    @param nu: coefficient for error lag
    @return: loss
    """
    if 'success' in candidate_data and candidate_data['success'] is False:
        logging.debug('discarding bad solver solution')
        return 9.001e9
    e_f = np.power((true_data['burst_frequency_mean'] - candidate_data['burst_frequency_mean']) * 1e3, 2)
    # print(f"true: {true_data['burst_frequency_mean']}; candidate: {candidate_data['burst_frequency_mean']}")
    e_dc = np.power(true_data['duty_cycle_mean'] - candidate_data['duty_cycle_mean'], 2)
    # print(f"true: {true_data['duty_cycle_mean']}; candidate: {candidate_data['duty_cycle_mean']}")
    # e_f = np.power((0.001 - candidate_data['burst_frequency_mean']) * 1e3, 2)
    # e_dc = np.power(0.2 - candidate_data['duty_cycle_mean'], 2)

    # e_mid = np.sum([np.power(nspike_mid - candidate_data['spike_per_burst'][:nspike_mid.size], 2)
    #                 for nspike_mid in candidate_data['spike_per_burst_mid']])
    # e_mid = np.sum(
    #     [np.power(nspike_mid[:np.min([nspike_mid.size, candidate_data['spike_per_burst'].size])] -
    #               candidate_data['spike_per_burst'][:np.min([nspike_mid.size, candidate_data['spike_per_burst'].size])],
    #               2).sum()
    #      for nspike_mid in candidate_data['spike_per_burst_mid']])

    # TODO: use log-barrier function instead of if, for both e_mid and num_bursts
    # if e_mid == 1.0:
    #     e_mid = 0.0
    e_sw = np.power(candidate_data['num_sw'], 2)
    e_lag = candidate_data['e_lag']
    return alpha * e_f + beta * e_dc + delta * e_sw + nu * e_lag  # + gamma * e_mid


def calc_mse_loss(true_data: dict, candidate_data: dict) -> float:
    """
    calculate mean squared error loss

    @param true_data: true data
    @param candidate_data: estimated data
    @return:loss
    """
    return np.square(true_data['y'] - candidate_data['y']).mean()


def calc_alonso_marder_loss_overall(true_data: dict,  # dict of currents, then
                                    candidate_data: dict,
                                    print_losses: bool = False) -> float:
    """
    calculate max overall loss from 5 injected currents

    @param true_data: true data
    @param candidate_data: estimated data
    @param print_losses: choose whether to print logging info
    @return: overall loss
    """
    total_loss = [calc_alonso_marder_loss(true_data[current_used], candidate_data[current_used])
                  for current_used in candidate_data.keys()]
    # return np.sum(total_loss)
    if print_losses:
        _ = [logging.info(f'TRUE {list(candidate_data.keys())[idx]} loss: {loss}') for idx, loss in enumerate(total_loss)]
    return np.max(total_loss)


def calc_mse_loss_overall(true_data: dict,  # dict of currents, then
                          candidate_data: dict) -> float:
    """
    calculate max overall loss from 5 injected currents

    @param true_data: true data
    @param candidate_data: estimated data
    @return: overall loss
    """
    total_loss = [calc_mse_loss(true_data[current_used], candidate_data[current_used])
                  for current_used in candidate_data.keys()]
    # return np.sum(total_loss)
    return np.max(total_loss)


def easy_neg_log_likelihood(sample_params: np.ndarray, true_data: dict,
                            sigma_d: float, random_seed: int,
                            loss_fn: str) -> Union[float, np.ndarray]:
    """
    likelihood function to compare actual vs estimated params with negative log

    :param sigma_d: standard deviation
    :param sample_params: estimated parameters
    :param true_data: true metric
    :param random_seed: the random number generator random seed
    :param loss_fn: loss function, either 'mse' or 'AlonsoMarder'
    :return: loss
    """
    # # MULTI-PROCESSED
    # with mp.get_context('spawn').Pool(len(list(true_data.keys()))) as pool:
    #     currents_used = true_data.keys()
    #     cached_generated_fn = MEM_CACHE.cache(generated_fn, verbose=0)
    #     candidate_model_outputs = pool.starmap(
    #         cached_generated_fn,
    #         [(sample_params, true_data[current_used]['t_step'],
    #           true_data[current_used]['t_end'], current_used) for current_used in currents_used])
    #     candidate_model_output = {current: outputs for current, outputs in zip(currents_used, candidate_model_outputs)}

    # SINGLE-PROCESSED
    currents_used = [0.30000000000000004]
    cached_generated_fn = MEM_CACHE.cache(generated_fn, verbose=0)
    candidate_model_outputs = starmap(
        cached_generated_fn,
        [(sample_params, true_data[current_used]['t_step'],
          true_data[current_used]['t_end'], current_used) for current_used in currents_used])
    candidate_model_output = {current: outputs for current, outputs in zip(currents_used, candidate_model_outputs)}

    if loss_fn == 'AlonsoMarder':
        loss = np.clip(np.nan_to_num(calc_alonso_marder_loss_overall(true_data, candidate_model_output),
                                     nan=9.001e8),
                       0.0, 9.001e8)  # + the_rng.normal(0, 1) * 1e3)
    else:
        loss = np.nan_to_num(calc_mse_loss_overall(true_data, candidate_model_output),
                             nan=9.001e8)  # + the_rng.normal(0, 1) * 1e3)

    if loss >= 9e8:
        logging.debug('discarding bad solver solution')
    elif loss < 1.0:
        logging.info(f'Reached loss {loss} using {sample_params}')
    return loss

    # # discard unstable solutions
    # if np.std(candidate_model_output['burst_frequency']) >= candidate_model_output['burst_frequency_mean'] * 0.1 or \
    #         np.std(candidate_model_output['duty_cycle']) >= candidate_model_output['duty_cycle_mean'] * 0.2:
    #     print('discarding unstable solution',
    #           np.std(candidate_model_output['burst_frequency']) >= candidate_model_output['burst_frequency_mean'] * 0.1,
    #           np.std(candidate_model_output['duty_cycle']) >= candidate_model_output['duty_cycle_mean'] * 0.2)
    #     return 9.0010e20 + the_rng.normal(0, 1) * 1e3
    # else:
    #     return calc_alonso_marder_loss(true_data, candidate_model_output) * sigma_d


def easy_neg_log_prior(num_params: int, lb_param: np.ndarray, ub_param: np.ndarray) -> pypesto.objective.NegLogParameterPriors:
    """
    prior distribution or prior knowledge about parameters

    @param num_params: number of parameters
    :param ub_param:
    :param lb_param:
    @return: negative log prior
    """

    # NOTE: the prior's default is log(max(parameter_prior_dict)), so default prior value is log(10000.)
    prior_list = []
    for i in range(num_params):
        prior_list.append(get_parameter_prior_dict(i, 'uniform', [0.0, 10000.0]))
    # create the prior
    neg_log_prior = pypesto.objective.NegLogParameterPriors(prior_list)
    return neg_log_prior


def clear_t_y(model_output_dict: Dict[str, Dict]):
    for model in model_output_dict.keys():
        for current in model_output_dict[model].keys():
            model_output_dict[model][current]['t_step'] = model_output_dict[model][current]['t'][1] - model_output_dict[model][current]['t'][0]
            model_output_dict[model][current]['t_end'] = model_output_dict[model][current]['t'][-1]
            del model_output_dict[model][current]['t']
            del model_output_dict[model][current]['y']
    return model_output_dict


def run_simulation(sigma_d=0.1, num_chains=12, n_iterations=600, loss='AlonsoMarder') -> sample.sample:
    """
    starts simulation

    @param sigma_d: standard deviation
    @param num_chains: number of chains
    @param n_iterations: number of iterations
    @param loss: loss function type
    @return: sampler result
    """
    model_output = pd.read_pickle('AlonsoMarderModel_generated_data.pkl', compression='infer')
    model_output = clear_t_y(model_output)
    model_output_a = model_output['a']

    logging.info(f'model loss (AlonsoMarder): {calc_alonso_marder_loss_overall(model_output_a, model_output_a, True)}')
    # initialization
    dim_full = 9  # number of parameters to solve for
    logging.info(f"iterations: {n_iterations}; num_chains: {num_chains}")
    lb = 1e-2
    ub = 2e3
    lb_param = np.array([800., 1., 1., 1.00, 1., 1., 0.01, 0.01, 100.])
    ub_param = np.array([2000., 10., 15., 200., 20., 200., 0.5, 0.5, 900.])

    random_seed = 7
    the_rng = np.random.default_rng(seed=random_seed)

    likelihood = pypesto.Objective(easy_neg_log_likelihood,
                                   fun_args=(model_output_a, sigma_d, random_seed, loss))

    prior_term = easy_neg_log_prior(dim_full, lb_param, ub_param)
    objective1 = pypesto.objective.AggregatedObjective([likelihood, prior_term])  # TODO: look at prior term

    problem = pypesto.Problem(objective=objective1, lb=lb_param, ub=ub_param)
    test = problem.objective(np.array([1169.641161, 7.617402, 10.412423, 146.423677, 13.193662, 72.802222, 0.15638, 0.064663, 648.934408]))
    logging.info(f'check that this number (bad params loss) is positive: {test}')
    logging.info('finished problem setup')

    with Pool(num_chains) as pool:
        sampler = sample.PoolAdaptiveParallelTemperingSampler(
            internal_sampler=sample.AdaptiveMetropolisSampler(),
            n_chains=num_chains,
            parallel_pool=pool
        )
        logging.info('finished sampler setup')

        x0 = [np.array([the_rng.uniform(low=lb_param[i], high=ub_param[i]) for i in range(dim_full)])
              for _ in range(num_chains)]
        logging.info('starting sampler')
        try:
            result_sampler = sample.sample(problem, n_iterations,
                                           sampler, x0=x0)
            sample.geweke_test(result=result_sampler)
            logging.info('finished result sampler')
            save_estimated_data(result_sampler)
            logging.info('saved data')
            return result_sampler
        except Exception as e:
            logging.exception(e)
            dump_filename = datetime.datetime.now().strftime('%Y-%b-%d_%H-%M.error.pkl')
            pickle.dump(sampler, open(dump_filename, "wb"))
            logging.info(f'dumped result_sampler to: {dump_filename}')
            raise e


def generate_summary_dataframe(sample_result: pypesto.sample.McmcPtResult,
                               remove_burn_in=True, is_manifold=False) -> pd.DataFrame:
    """
    generate sorted dataframe with/without burn in

    @param sample_result: sample results
    @param remove_burn_in: chose to remove burn in
    @return: sorted data
    """
    all_x = sample_result['trace_x']
    all_post = sample_result['trace_neglogpost']
    if remove_burn_in:
        full_x = np.empty((0, all_x.shape[2]))
        full_post = np.empty((0, ))
        for chain_id, fp32_burn_in_idx in enumerate(sample_result['burn_in']):
            burn_in_idx = int(fp32_burn_in_idx)
            full_x = np.concatenate((full_x, all_x[chain_id, burn_in_idx:, :]))
            full_post = np.concatenate((full_post, all_post[chain_id, burn_in_idx:]))
    else:
        full_x = all_x.reshape((all_x.shape[0] * all_x.shape[1], all_x.shape[2]))
        full_post = all_post.ravel()
    full_sort_idx = np.argsort(full_post)
    full_x_sort = full_x[full_sort_idx]
    # grabs optimal params. change fullsortidx to look at "less optimal" params
    # opt_param = full_x[full_sort_idx[0]]
    if is_manifold:
        return pd.DataFrame.from_dict({
            'neglogpost': full_post[full_sort_idx],
            'g_Na': full_x_sort[:, 0],
            'g_Kd': full_x_sort[:, 1],
        })
    return pd.DataFrame.from_dict({
        'neglogpost': full_post[full_sort_idx],
        'g_Na': full_x_sort[:, 0],
        'g_CaT': full_x_sort[:, 1],
        'g_CaS': full_x_sort[:, 2],
        'g_A': full_x_sort[:, 3],
        'g_KCa': full_x_sort[:, 4],
        'g_Kd': full_x_sort[:, 5],
        'g_H': full_x_sort[:, 6],
        'g_L': full_x_sort[:, 7],
        'tau_ca': full_x_sort[:, 8]
    })


def save_estimated_data(result_sampler: pypesto.Result) -> None:
    """
    save estimated data in hdf5

    @param result_sampler: sample results
    @return: None
    """
    result_sampler_dict = result_sampler.sample_result
    result_sampler_dict.pop('auto_correlation')
    result_sampler_dict.pop('message')
    result_sampler_dict.pop('effective_sample_size')
    with h5py.File('AlonsoMarderModel_estimated_data.hdf5', 'w') as le_file:
        for key, value in result_sampler_dict.items():
            le_file.create_dataset(key, data=value)
    # Note: how to open file example - list(F['trace_x'])
    return None


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    result = run_simulation(n_iterations=14000, num_chains=40, loss='AlonsoMarder')
    # df = generate_summary_dataframe(result.sample_result)
    # now = datetime.datetime.now()
    # df.to_hdf('mcmc_results.h5', key=f'{now:result_%Y_%m_%d__%H_%M}')
    dump_filename = datetime.datetime.now().strftime('%Y-%b-%d_%H-%M.pypesto_results.pkl')
    pickle.dump(result.sample_result, open(dump_filename, "wb"))