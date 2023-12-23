import os

os.environ['MKL_NUM_THREADS'] = "1"
from typing import Dict, Union
from AlonsoMarderModel import AlonsoMarderModel
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

models = {
    'a': {
        'conductances': {
            'g_Na': 1076.392,  # uS, transient sodium conductance
            'g_CaT': 6.4056,  # uS, low-threshold calcium conductance
            'g_CaS': 10.048,  # uS, slow calcium conductance
            'g_A': 8.0384,  # uS, transient potassium conductance
            'g_KCa': 17.584,  # uS, calcium-dependent potassium conductance
            'g_Kd': 124.0928,  # uS, potassium conductance
            'g_H': 0.11304,  # uS, hyperpolarization-activated cation conductance
            'g_L': 0.17584,  # uS, leak conductance
        },
        'tau_ca': 653.5
    },
    # add any other models desired here
}


def make_dict_model_from_mcmc_result_list(
        row_values: np.ndarray, name: str = 'mcmc') -> Dict[str, Dict[str, Union[Dict[str, float], float]]]:
    """
    makes dictionary model from mcmc results

    @param row_values: initial row vector from MCMC sampler results
    @param name: name of the model
    @return: model
    """
    return {
        name: {
            'conductances': {
                'g_Na': row_values[1],  # uS, transient sodium conductance
                'g_CaT': row_values[2],  # uS, low-threshold calcium conductance
                'g_CaS': row_values[3],  # uS, slow calcium conductance
                'g_A': row_values[4],  # uS, transient potassium conductance
                'g_KCa': row_values[5],  # uS, calcium-dependent potassium conductance
                'g_Kd': row_values[6],  # uS, potassium conductance
                'g_H': row_values[7],  # uS, hyperpolarization-activated cation conductance
                'g_L': row_values[8],  # uS, leak conductance
            },
            'tau_ca': row_values[9]
        }
    }


def plot_model_outputs(
        key: str, dict_models: dict, current_injected: float = 0.0, silent: bool = False) -> np.array:
    """
    plot all 6 models mV vs ms with its threshold

    :param key: model type (a through f)
    :param dict_models: model specification (a through f)
    :param current_injected: amount of current to inject
    :param silent: False by default to show plots
    :return: time steps, voltage trace, spike times and threshold and their respective plots
    """
    print(f'key : {key}')
    time_in_seconds = 6.0
    the_model = AlonsoMarderModel(injected_current=current_injected,
                                  conductances=dict_models[key]['conductances'],
                                  tau_ca=dict_models[key]['tau_ca'])
    time_steps = np.arange(0.0, time_in_seconds * 1e3, 0.01)
    start = time.time()
    model_output = the_model.run_simulation(time_steps)
    print(f'compute time: {time.time() - start}')
    if not silent:
        plt.plot(model_output["t"], model_output["y"])
        # plt.plot(model_output["spike_times"], model_output["spike_threshold"], "ro")
        plt.xlabel('time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.title(f'model: {key}')
        # plt.legend(the_model.get_state_vars_labels())
        plt.show()
    return model_output


def plot_model_comparison(dict_models: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> None:
    for key, val in dict_models.items():
        if key == 'a':
            continue
        for current_index in np.arange(0, .5, .1):
            fig, ((g), (h)) = plt.subplots(2)
            fig.suptitle(f'ground truth model (top) and mcmc (bot) - model: {key} current: {float(current_index)}')
            g.plot(dict_models['a'][current_index]['t'], dict_models['a'][current_index]['y'], 'tab:orange')
            h.plot(dict_models[key][current_index]['t'], dict_models[key][current_index]['y'])
            plt.xlabel('samples')
            for ax in fig.get_axes():
                ax.label_outer()
            plt.savefig(f'{current_index}.png')
            plt.show()
    return None


def convert_dict_of_lists_to_model(
        model_dict: Dict[str, Union[np.ndarray, list]]) -> Dict[str, Dict[str, Union[Dict[str, float], float]]]:
    all_models = {}
    for key, val in model_dict.items():
        model = {
            key: {
                'conductances': {
                    'g_Na': val[0],  # uS, transient sodium conductance
                    'g_CaT': val[1],  # uS, low-threshold calcium conductance
                    'g_CaS': val[2],  # uS, slow calcium conductance
                    'g_A': val[3],  # uS, transient potassium conductance
                    'g_KCa': val[4],  # uS, calcium-dependent potassium conductance
                    'g_Kd': val[5],  # uS, potassium conductance
                    'g_H': val[6],  # uS, hyperpolarization-activated cation conductance
                    'g_L': val[7],  # uS, leak conductance
                },
                'tau_ca': val[8]
            }
        }
        all_models.update(model)
    return all_models


def compute_and_save_models(
        models_to_make: Dict[str, Dict[str, Union[Dict[str, float], float]]],
        mcmc_model_desired, compute_single_current=False) -> None:
    """
    Store all models in a pickle

    :return: None
    """
    model = {}
    model_all = {}
    if compute_single_current:
        for key in mcmc_model_desired:
            if key == 'all':
                model_all.update({current_to_plot: plot_model_outputs(key, models_to_make, current_to_plot, False) for
                                  current_to_plot in np.arange(0.0, 0.5, 0.1)})
            else:
                model.update({float(key): plot_model_outputs(key, models_to_make, float(key), False)})

        model_outs = {
            'a':
                {current_to_plop: plot_model_outputs('a', models_to_make, current_to_plop, False)
                 for current_to_plop in np.arange(0.0, 0.5, 0.1)},
            'mcmc_single_currents': model,
            'mcmc_all_current': model_all,
        }

    else:
        model_outs = {
            key: {current_to_plop: plot_model_outputs(key, models_to_make, current_to_plop, current_to_plop != 0.0)
                  for current_to_plop in np.arange(0.0, 0.5, 0.1)}
            for key in models_to_make.keys()}
    # compare_models(model_outs)
    plot_model_comparison(model_outs)
    with open('AlonsoMarderModel_generated_data.pkl', 'wb') as filehandle:
        pickle.dump(model_outs, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
    return None


if __name__ == '__main__':
    # plot_model_outputs()
    mcmc_model_dict = {
        '0.0': [1.36504984e+03, 6.90471282e+00, 1.03010743e+01, 7.52944902e+01, 1.79958197e+01, 1.07390603e+02,
                4.15844030e-01, 1.75782751e-01, 7.38407016e+02],  # new uniform, 82.89
        '0.1': [1510.54096, 7.84354, 10.96749, 54.15303, 16.48460, 139.69044, 0.27945, 0.19041, 607.57802],  # 0.00212
        '0.2': [1823.57888, 6.95997, 13.63717, 145.06658, 13.45328, 92.24636, 0.35482, 0.18666, 342.65233],  # 0.02280
        '0.30000000000000004': [1554.45888, 6.53608, 14.23725, 140.65041, 12.56214, 53.72467, 0.21759, 0.14271,
                                470.50603],  # 0.00189
        '0.4': [1326.94727, 7.60586, 8.80298, 35.66033, 10.20653, 91.64060, 0.30257, 0.21334, 238.34122],  # 0.00439
        'all': [1307.70985, 8.85968, 13.41245, 114.53560, 16.89606, 121.59780, 0.28586, 0.12557, 794.03210],  # 18.76235
    }

    mcmc_model = convert_dict_of_lists_to_model(mcmc_model_dict)
    models.update(mcmc_model)
    # plot_model_comparison(models)
    compute_and_save_models(models, mcmc_model_dict, compute_single_current=True)
    model_outputs = pickle.load(open('AlonsoMarderModel_generated_data.pkl', 'rb'))
