import os
os.environ['MKL_NUM_THREADS'] = "1"
import logging
from typing import List, Tuple, Dict, Union
import numpy as np
from collections import OrderedDict
from scipy.integrate import solve_ivp
import time


class AlonsoMarderModel(object):
    """
    AlonsoMarderModel provides a class for the AlonsoMarder neuronal modal provided in:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6395073/

    """

    def __init__(self, injected_current=None, initial_conditions=None, reversal_potentials=None, conductances=None,
                 inf_alphas=None, inf_betas=None, tau_cs=None, tau_ds=None, tau_as=None, tau_bs=None,
                 tau_a2s=None, tau_b2s=None, tau_ca=None, spike_threshold=None):
        self.voltage_trace = None
        self.spike_times = None
        self.time_steps = None
        self.constants = {
            'reftemp_celsius': 10.0,  # degC, reference temperature
            'temp_celsius': 10.0,  # degC,  temperature
            'gas_constant': 8.314 * pow(10, 3),  # Ideal Gas Constant (*10^3 to put into mV)
            'base_temp_kelvin': 273.15,  # Temperature in Kelvin
            'z': 2.0,  # Valence of Caclium Ions
            'faraday': 96485.33,  # Faraday's Constant
            'ca_conc_extracellular': 3000.0,  # Outer Ca Concentration (uM)
            'ca_conv_factor': 0.94,  # Outer Ca Concentration (uM)
            'ca_conc_background': 0.05,  # Outer Ca Concentration (uM)
            'tau_ca_conc_intracellular': tau_ca or 6.535e2,  # Outer Ca Concentration (uM)
            'C': 10.0,  # nF / cm^2, membrane capacitance
            'I_app': injected_current or 0.0,  # nA, externally-applied current
        }
        self.initial_conditions = initial_conditions or OrderedDict({
            'V': -51.,  # mV, membrane voltage
            'm_Na': 0.,  # sodium activation variable
            'h_Na': 0.,  # sodium inactivation variable
            'm_CaT': 0.,  # low-threshold calcium activation variable
            'h_CaT': 0.,  # low-threshold calcium inactivation variable
            'm_CaS': 0.,  # slow calcium activation variable
            'h_CaS': 0.,  # slow calcium inactivation variable
            'm_H': 0.,  # hyperpolarization-activated cation activation variable
            'h_H': 1.,  # hyperpolarization-activated cation inactivation variable
            'm_Kd': 0.,  # potassium activation variable
            'h_Kd': 1.,  # potassium inactivation variable
            'm_KCa': 0.,  # mV, calcium-dependent potassium activation variable
            'h_KCa': 1.,  # mV, calcium-dependent potassium inactivation variable
            'm_A': 0.,  # mV, transient potassium activation variable
            'h_A': 0.,  # mV, transient potassium inactivation variable
            'm_L': 1.,  # leak channel activation variable
            'h_L': 1.,  # leak channel inactivation variable
            'ca_conc_intracellular': 5.,  # uM, intracellular Ca concentration initial condition
        })
        self.spike_threshold = spike_threshold or {
            'spike_threshold': self.initial_conditions["V"] + 15.0,
            'threshold_spike': -15.,
            'threshold_mid_upper': -35.,
            'threshold_mid_lower': -45.,
            'threshold_slow_wave': -49.999,
        }
        self.reversal_potentials = reversal_potentials or {
            'E_L': -50.,  # mV, leak reversal potential
            'E_Na': 30.,  # mV, sodium reversal potential
            'E_CaT': self._calculate_calcium_rev_potential(
                self.initial_conditions['ca_conc_intracellular'],
                self.constants['temp_celsius']),  # mV, low-threshold calcium reversal potential
            'E_CaS': self._calculate_calcium_rev_potential(
                self.initial_conditions['ca_conc_intracellular'],
                self.constants['temp_celsius']),  # mV, slow calcium reversal potential
            'E_Kd': -80.,  # mV, potassium reversal potential
            'E_KCa': -80.,  # mV, calcium-dependent potassium reversal potential
            'E_A': -80.,  # mV, transient potassium reversal potential
            'E_H': -20.,  # mV, hyperpolarization-activated cation reversal potential
        }
        self.conductances = conductances or {
            'g_Na': 1.0764e3,  # uS, transient sodium conductance
            'g_CaT': 6.4056e0,  # uS, low-threshold calcium conductance
            'g_CaS': 1.0048e1,  # uS, slow calcium conductance
            'g_A': 8.0384e0,  # uS, transient potassium conductance
            'g_KCa': 1.7584e1,  # uS, calcium-dependent potassium conductance
            'g_Kd': 1.240928e2,  # uS, potassium conductance
            'g_H': 1.1304e-1,  # uS, hyperpolarization-activated cation conductance
            'g_L': 1.7584e-1,  # uS, leak conductance
        }
        self.inf_alphas = inf_alphas or {
            'm_Na': 25.5,  # sodium activation variable
            'h_Na': 48.9,  # sodium inactivation variable
            'm_CaT': 27.1,  # low-threshold calcium activation variable
            'h_CaT': 32.1,  # low-threshold calcium inactivation variable
            'm_CaS': 33.0,  # slow calcium activation variable
            'h_CaS': 60.0,  # slow calcium inactivation variable
            'm_H': 70.0,  # hyperpolarization-activated cation activation variable
            'm_Kd': 12.3,  # potassium activation variable
            'm_KCa': 28.3,  # mV, calcium-dependent potassium activation variable
            'm_A': 27.2,  # mV, transient potassium activation variable
            'h_A': 56.9,  # mV, transient potassium inactivation variable
        }
        self.inf_betas = inf_betas or {
            'm_Na': -5.29,  # sodium activation variable
            'h_Na': 5.18,  # sodium inactivation variable
            'm_CaT': -7.20,  # low-threshold calcium activation variable
            'h_CaT': 5.50,  # low-threshold calcium inactivation variable
            'm_CaS': -8.1,  # slow calcium activation variable
            'h_CaS': 6.20,  # slow calcium inactivation variable
            'm_H': 6.0,  # hyperpolarization-activated cation activation variable
            'm_Kd': -11.8,  # potassium activation variable
            'm_KCa': -12.6,  # mV, calcium-dependent potassium activation variable
            'm_A': -8.70,  # mV, transient potassium activation variable
            'h_A': 4.90,  # mV, transient potassium inactivation variable
        }
        self.tau_cs = tau_cs or {
            'm_Na': 1.32,  # sodium activation variable
            'h_Na_0': 0.0,  # sodium inactivation variable
            'h_Na_1': 1.50,  # sodium inactivation variable
            'm_CaT': 21.7,  # low-threshold calcium activation variable
            'h_CaT': 105.0,  # low-threshold calcium inactivation variable
            'm_CaS': 1.40,  # slow calcium activation variable
            'h_CaS': 60.0,  # slow calcium inactivation variable
            'm_H': 272.0,  # hyperpolarization-activated cation activation variable
            'm_Kd': 7.20,  # potassium activation variable
            'm_KCa': 90.3,  # mV, calcium-dependent potassium activation variable
            'm_A': 11.6,  # mV, transient potassium activation variable
            'h_A': 38.6,  # mV, transient potassium inactivation variable
        }
        self.tau_ds = tau_ds or {
            'm_Na': 1.26,  # sodium activation variable
            'h_Na_0': -0.67,  # sodium inactivation variable
            'h_Na_1': -1.00,  # sodium inactivation variable
            'm_CaT': 21.3,  # low-threshold calcium activation variable
            'h_CaT': 89.8,  # low-threshold calcium inactivation variable
            'm_CaS': 7.00,  # slow calcium activation variable
            'h_CaS': 150.0,  # slow calcium inactivation variable
            'm_H': -1499.0,  # hyperpolarization-activated cation activation variable
            'm_Kd': 6.40,  # potassium activation variable
            'm_KCa': 75.1,  # mV, calcium-dependent potassium activation variable
            'm_A': 10.4,  # mV, transient potassium activation variable
            'h_A': 29.2,  # mV, transient potassium inactivation variable
        }
        self.tau_as = tau_as or {
            'm_Na': 120.0,  # sodium activation variable
            'h_Na_0': 62.9,  # sodium inactivation variable
            'h_Na_1': 34.9,  # sodium inactivation variable
            'm_CaT': 68.1,  # low-threshold calcium activation variable
            'h_CaT': 55.0,  # low-threshold calcium inactivation variable
            'm_CaS': 27.0,  # slow calcium activation variable
            'h_CaS': 55.0,  # slow calcium inactivation variable
            'm_H': 42.2,  # hyperpolarization-activated cation activation variable
            'm_Kd': 28.3,  # potassium activation variable
            'm_KCa': 46.0,  # mV, calcium-dependent potassium activation variable
            'm_A': 32.9,  # mV, transient potassium activation variable
            'h_A': 38.9,  # mV, transient potassium inactivation variable
        }
        self.tau_bs = tau_bs or {
            'm_Na': -25.0,  # sodium activation variable
            'h_Na_0': -10.0,  # sodium inactivation variable
            'h_Na_1': 3.60,  # sodium inactivation variable
            'm_CaT': -20.5,  # low-threshold calcium activation variable
            'h_CaT': -16.9,  # low-threshold calcium inactivation variable
            'm_CaS': 10.0,  # slow calcium activation variable
            'h_CaS': 9.00,  # slow calcium inactivation variable
            'm_H': -8.73,  # hyperpolarization-activated cation activation variable
            'm_Kd': -19.2,  # potassium activation variable
            'm_KCa': -22.7,  # mV, calcium-dependent potassium activation variable
            'm_A': -15.2,  # mV, transient potassium activation variable
            'h_A': -26.5,  # mV, transient potassium inactivation variable
        }
        self.tau_a2s = tau_a2s or {
            'm_CaS': 70.0,
            'h_CaS': 65.0,
        }
        self.tau_b2s = tau_b2s or {
            'm_CaS': -13.0,
            'h_CaS': -16.0,
        }
        self.q10s = {
            'i_Na': 3.,
            'i_CaT': 3.,
            'i_CaS': 3.,
            'i_H': 1.,
            'i_Kd': 4.,
            'i_KCa': 4.,
            'i_A': 3.,
            'i_L': 1.,
            'g_Na': 1.,
            'm_Na': 1.,
            'h_Na': 1.,
            'g_CaT': 1.,
            'm_CaT': 1.,
            'h_CaT': 1.,
            'g_CaS': 1.,
            'm_CaS': 1.,
            'h_CaS': 1.,
            'g_A': 1.,
            'm_A': 1.,
            'h_A': 1.,
            'g_KCa': 1.,
            'm_KCa': 1.,
            'h_KCa': 1.,
            'g_Kd': 1.,
            'm_Kd': 1.,
            'h_Kd': 1.,
            'g_H': 1.,
            'm_H': 1.,
            'h_H': 1.,
            'g_L': 1.,
            'tau_Ca': 1.,
        }
        self.channel_types = ('Na', 'CaT', 'CaS', 'H', 'Kd', 'KCa', 'A', 'L')
        self.channel_currents = {}
        self.state_vars_constant = ('m_L', 'h_H', 'h_Kd', 'h_KCa', 'h_L')
        self.state_vars_labels = self.get_state_vars_labels()
        self.state_vars = [self.initial_conditions[key] for key in self.state_vars_labels]

    def get_state_var(self, key: str) -> float:
        """
        get a state variable

        :param key: name of channel state var
        :return: the current value of the given state variable
        """
        if key in self.state_vars_constant:
            return 1.0
        index_state_var = self.state_vars_labels.index(key)
        return self.state_vars[index_state_var]

    def get_state_vars_labels(self) -> list:
        """
        get state variable labels

        :return: state variable labels
        """
        state_vars_m = [f'm_{ch}' for ch in self.channel_types]
        [state_vars_m.remove(x) for x in self.state_vars_constant if x in state_vars_m]
        state_vars_h = [f'h_{ch}' for ch in self.channel_types]
        [state_vars_h.remove(x) for x in self.state_vars_constant if x in state_vars_h]
        return ['V', 'ca_conc_intracellular'] + state_vars_m + state_vars_h

    def _calculate_calcium_rev_potential(self, ca_conc_intracellular, temp_celsius):
        """
        computed dynamically using the Nernst equation assuming an extra-cellular calcium concentration of 3e3 uMolars.

        :param ca_conc_intracellular: calcium intracellular concentration
        :param temp_celsius: temperature in c
        :return: calcium reverse potential
        """
        rtzf_term = self.constants['gas_constant'] * (self.constants['base_temp_kelvin'] + temp_celsius)
        rtzf_term /= (self.constants['z'] * self.constants['faraday'])
        return rtzf_term * np.log10(self.constants['ca_conc_extracellular'] / ca_conc_intracellular)

    def _calculate_normal_inf_response(self, key: str, voltage: float) -> float:
        """
        calculates infinite response

        :param key: name of the channel's state variables
        :param voltage: current voltage
        :return: normal infinite response
        """
        return 1. / (1. + np.exp((voltage + self.inf_alphas[key]) / self.inf_betas[key]))

    def _calculate_kca_inf_response(self, key: str, voltage: float) -> float:
        """
        calculates infinite response with respect to KCa

        :param key: name of channel's state variables
        :param voltage: current voltage
        :return: spec infinite response
        """
        index_of_conc_kca = self.state_vars_labels.index('ca_conc_intracellular')
        conc_kca = self.state_vars[index_of_conc_kca]
        left_term = conc_kca / (conc_kca + 3.0)
        return left_term / (1. + np.exp((voltage + self.inf_alphas[key]) / self.inf_betas[key]))

    def _calculate_inf_response(self, key: str, voltage: float) -> float:
        """
        compute the infinite response for a channel's state variable

        :param key: the name of a channel's state variable
        :param voltage: current voltage
        :return: the value of the channel's state variable's current infinite response
        """
        if 'KCa' in key:
            return self._calculate_kca_inf_response(key, voltage)
        else:
            return self._calculate_normal_inf_response(key, voltage)

    def get_current_voltage(self) -> float:
        """
        get value from state variable and return float

        :return: current voltage
        """
        index_voltage = self.state_vars_labels.index('V')
        return self.state_vars[index_voltage]

    def _calculate_normal_tau(self, key: str, voltage: float) -> float:
        """
        calcualtes tau normally by : CT - DT/(1. + exp((Volt + AT)/BT))

        :param key: name of channels state variables current voltage
        :param voltage: current voltage
        :return: normal tau
        """
        timeconst = self.tau_cs[key]
        timeconst -= self.tau_ds[key] / (1. + np.exp((voltage + self.tau_as[key]) / self.tau_bs[key]))
        return timeconst

    def _calculate_cas_tau(self, key: str, voltage: float) -> float:
        """
        calculates CaS tau different from normal tau by : CT + DT/(exp((Volt + AT)/BT) + exp((Volt + AT2)/BT2))

        :param key: name of channels state variables
        :param voltage: current voltage
        :return: spec tau
        """
        div_term = np.exp((voltage + self.tau_as[key]) / self.tau_bs[key])
        div_term += np.exp((voltage + self.tau_a2s[key]) / self.tau_b2s[key])
        return self.tau_cs[key] + self.tau_ds[key] / div_term

    def _calculate_double_tau(self, key: str, voltage: float) -> float:
        """
        calculate normal tau multiplied twice with different channel states

        :param key: name of channels state variables
        :param voltage: current voltage
        :return: double tau
        """
        total = self._calculate_normal_tau(f'{key}_0', voltage) * self._calculate_normal_tau(f'{key}_1', voltage)
        return total

    def _calculate_tau(self, key: str) -> float:
        """
        calculate the time constant for a channel's state variable

        :param key: the name of a channel's state variable
        :return: the value of the channel's state variable's current time constant
        """
        voltage = self.get_current_voltage()
        if 'h_Na' in key:
            return self._calculate_double_tau(key, voltage)
        elif 'CaS' in key:
            return self._calculate_cas_tau(key, voltage)
        else:
            return self._calculate_normal_tau(key, voltage)

    def scale_time(self, key: str, value_to_scale: float) -> float:
        """
        scare the input by using q10 corresponding to the key in the dict

        :param key: name of channels state variables
        :param value_to_scale: q10 scalar
        :return: scaled value with corresponding q10
        """
        temp = self.constants['temp_celsius']
        reftemp = self.constants['reftemp_celsius']
        return value_to_scale * pow(self.q10s[key], -(temp - reftemp) / 10.0)

    def _calculate_channel_current(self, channel: str) -> float:
        """
        calculates the channel current

        :param channel: ionic channel names
        :return: channel current
        """
        g = self.conductances[f'g_{channel}']
        e_rev = self.reversal_potentials[f'E_{channel}']
        h = self.get_state_var(f'h_{channel}')
        m = self.get_state_var(f'm_{channel}')
        voltage = self.get_current_voltage()
        q = self.q10s[f'i_{channel}']
        pow_term = pow(m, q)
        return g * pow_term * h * (voltage - e_rev)

    def _calculate_dvdt(self, channel_currents: dict) -> float:
        """
        calculates the derivative of the voltage wrt time

        :param channel_currents: channel currents
        :return: deriv of current
        """
        return (-sum(channel_currents.values()) + self.constants['I_app']) / self.constants['C']

    def _calculate_dstate_dt(self, key: str) -> float:
        """
        calculates the derivative of the states channels

        :param key: name of channel state variables
        :return: deriv of state variable
        """
        state_var = self.get_state_var(key)
        voltage = self.get_current_voltage()
        inf_state_var = self._calculate_inf_response(key, voltage)
        # tau_state_var = self.scale_time(f'tau_{key}', self._calculate_tau(key))
        tau_state_var = self._calculate_tau(key)
        return (inf_state_var - state_var) / tau_state_var

    def _calculate_dca_conc_intracellular_dt(self, channel_currents: dict) -> float:
        """
        calculates the derivative of calcium's intracellular concentration

        :param channel_currents: channel current
        :return: deriv of calcium's intracellular concentration
        """
        ca_conc_intra = self.get_state_var('ca_conc_intracellular')
        outcalc = -self.constants['ca_conv_factor']
        outcalc *= (channel_currents['CaT'] + channel_currents['CaS'])
        outcalc += self.constants['ca_conc_background'] - ca_conc_intra
        outcalc /= self.scale_time('tau_Ca', self.constants['tau_ca_conc_intracellular'])
        return outcalc

    def _calculate_dstate_variable(self, key, channel_currents: dict) -> float:
        """
        calculates state variables

        :param key: name of channel state variables
        :param channel_currents: channel current
        :return: state variables
        """
        if key == 'V':
            return self._calculate_dvdt(channel_currents)
        elif key == 'ca_conc_intracellular':
            return self._calculate_dca_conc_intracellular_dt(channel_currents)
        elif key in self.state_vars_constant:
            return np.float64(0.0)
        else:
            return self._calculate_dstate_dt(key)

    # noinspection PyUnusedLocal
    def update_state_variables(self, t: float, y: np.array) -> np.array:
        """
        updates state variables (dy/dt) function for solve_ivp or other ODE solver

        :param t: current time, ignored
        :param y: array of current values of state variables
        :return: updated state variables
        """
        self.state_vars = [y[self.state_vars_labels.index(key)]
                           for key in self.state_vars_labels]
        self._update_ca_rev_potential()
        self.channel_currents = {ch: self._calculate_channel_current(ch)
                                 for ch in self.channel_types}
        return np.array([self._calculate_dstate_variable(key, self.channel_currents)
                         for key in self.state_vars_labels])
        # return np.array(new_dstate_vars)
        # new_array = np.array(new_dstate_vars)
        # if np.all(np.isfinite(new_array)):
        #     return new_array
        # else:
        #     return np.ones_like(new_array) + np.nan

    def get_state_vars_and_labels(self) -> Tuple[List[float], List[str]]:
        """
        Get the current values of the state variables and their labels

        :return: a Tuple of: a list of the values of the state variables, and a list of the labels
        """
        return self.state_vars, self.state_vars_labels

    def get_initial_conditions(self) -> Tuple[List[float], List[str]]:
        """
        Get the initial conditions

        :return: a Tuple of: a list of the values of the state variables, and a list of the labels
        """
        return [self.initial_conditions[key] for key in self.state_vars_labels], self.state_vars_labels

    def _update_ca_rev_potential(self) -> None:
        """
        updates calcium reverse potential

        :return: None
        """
        new_ca_rev_potential = self._calculate_calcium_rev_potential(
            self.get_state_var('ca_conc_intracellular'),
            self.constants['temp_celsius'])
        self.reversal_potentials['E_CaT'] = new_ca_rev_potential  # mV, low-threshold calcium reversal potential
        self.reversal_potentials['E_CaS'] = new_ca_rev_potential  # mV, slow calcium reversal potential

    def run_simulation(self, time_steps: np.ndarray) -> Dict[str, Union[np.ndarray, bool, float]]:
        """
        run simulation of model
        :param time_steps: time steps
        :return: 't': time steps, 'y': voltage trace, and 'spike_times': spike times
        """
        self.time_steps = time_steps
        init_cond = self.get_initial_conditions()[0]
        # time_start = time.time()
        sol = solve_ivp(self.update_state_variables, [self.time_steps[0],
                                                      self.time_steps[-1]],
                        init_cond, "BDF", self.time_steps)
        # out_time = time.time() - time_start
        # logging.debug(f'AMM: compute time: {out_time} s')
        self.voltage_trace = sol.y[0]
        logging.debug("AMM: computing trace characteristics")
        dict_characteristics = self.convert_trace_to_spike_characteristics_tonic(self.voltage_trace,
                                                                                 self.time_steps,
                                                                                 self.spike_threshold)
        logging.debug("AMM: returning")
        return {"t": self.time_steps,
                "y": self.voltage_trace,
                **dict_characteristics}

    @staticmethod
    def convert_trace_to_spike_times_upward(voltage: np.array, times: np.array, threshold: float) -> np.ndarray:
        """
        Converts voltage traces to spike times
        :param voltage: array of voltages (mV)
        :param times:    array of time values (.1ms)
        :param threshold:   threshold for spike activation
        :return: an array of spike times
        """
        # grab all indices above the activation threshold0
        voltage_indices_ge_th = np.where(voltage >= threshold)[0]
        voltage_indices_ge_th = voltage_indices_ge_th[np.where(voltage_indices_ge_th < (len(times) - 1))[0]]
        # only grab the index that is directly below the threshold and remove all other indexes
        voltage_indeces_upward_th = voltage_indices_ge_th[np.where(voltage[voltage_indices_ge_th - 1] < threshold)]
        # use index from voltage and use for time
        return times[voltage_indeces_upward_th]

    @staticmethod
    def convert_trace_to_spike_times_downward(voltage: np.array, times: np.array, threshold: float) -> np.ndarray:
        """
        Converts voltage traces to spike times
        :param voltage: array of voltages (mV)
        :param times:    array of time values (.1ms)
        :param threshold:   threshold for spike activation
        :return: an array of spike times
        """
        # grab all indices above the activation threshold0
        voltage_indices_ge_th = np.where(voltage >= threshold)[0]
        voltage_indices_ge_th = voltage_indices_ge_th[np.where(voltage_indices_ge_th < (len(times) - 1))[0]]
        # only grab the index that is directly below the threshold and remove all other indexes
        voltage_indeces_downward_th = voltage_indices_ge_th[np.where(voltage[voltage_indices_ge_th + 1] < threshold)]
        # use index from voltage and use for time
        return times[voltage_indeces_downward_th]

    @staticmethod
    def calculate_num_spikes_per_burst(spike_times, temporal_interval: Union[int, float]):
        spike_diff = np.diff(spike_times)
        spike_gt_temp = np.where(np.array(spike_diff) > temporal_interval)[0]
        # burst_end_loc = spike_gt_temp.astype(np.int) + 1
        burst_end_loc_append = np.append(spike_gt_temp, len(spike_times) - 1)
        return np.diff(np.insert(burst_end_loc_append, 0, 0))

    @staticmethod
    def clean_up_num_spikes_per_burst(num_spikes: np.ndarray) -> np.ndarray:
        # indices_of_trailing_downward = np.where(num_spikes == 1)[0] - 1
        # num_spikes[indices_of_trailing_downward] += 1
        return num_spikes[num_spikes > 1]

    @staticmethod
    def calculate_num_interbursts(spike_times, temporal_interval):
        # spike_diff = [(spike_times[x + 1] - spike_times[x]) for x in range(len(spike_times) - 1)]
        spike_diff = np.diff(spike_times)
        spike_gt_temp = np.where(np.array(spike_diff) > temporal_interval)[0]
        burst_end_loc = spike_gt_temp.astype(np.int) + 1
        burst_end_loc_append = np.append(spike_gt_temp, len(spike_times)-1)
        burst_start_loc_append = np.insert(burst_end_loc, 0, 0)

        return {'bs_loc': burst_start_loc_append,
                'be_loc': burst_end_loc_append
                }

    def convert_trace_to_spike_characteristics_tonic(self, voltage: np.ndarray, times: np.ndarray,
                                                     threshold: dict) -> Dict[str, np.ndarray]:
        times_before_th_upward = self.convert_trace_to_spike_times_upward(voltage, times, threshold['threshold_spike'])
        times_before_th_downward = self.convert_trace_to_spike_times_downward(voltage, times,
                                                                              threshold['threshold_mid_upper'])
        times_before_th_downward2 = self.convert_trace_to_spike_times_downward(voltage, times,
                                                                               threshold['threshold_mid_lower'])
        e_lag = np.mean(np.abs(times_before_th_upward[:times_before_th_downward.size] - times_before_th_downward))
        t_sp = -20
        # check if a spike is within 100ms AFTER collecting all the spike times that crosses upward on a th.
        temporal_interval = 100.
        spike_times = times_before_th_upward
        num_bursts = self.calculate_num_interbursts(spike_times, temporal_interval)

        # TODO: use log-barrier function instead of if
        if spike_times.size < 1 or times_before_th_downward.size < 1 or times_before_th_downward2.size < 1 \
                or num_bursts['bs_loc'].size < 2 or num_bursts['be_loc'].size < 2:
            return {'burst_frequency_mean': 1e5,
                    'duty_cycle_mean': 1e5,
                    'times_before_th_downward': times_before_th_downward,
                    'times_before_th_downward2': times_before_th_downward2,
                    'num_sw': 0.0,
                    'e_lag': 0.0,
                    'num_spikes': np.array([]),
                    'num_mid': np.array([]),
                    'spike_per_burst': np.array([]),
                    'spike_per_burst_mid': np.array([]),
                    }

        bs = spike_times[num_bursts['bs_loc']]
        be = spike_times[num_bursts['be_loc']]

        if np.all(bs[-1] == be[-1]):
            bs = bs[:-1]
            be = be[:-1]

        burst_duration = be - bs  # [:be.size]
        # period_half = be
        # period = burst_duration + period_half
        period = np.diff(bs)
        burst_frequency = 1 / np.diff(be)
        duty_cycle = burst_duration[:period.size] / np.diff(bs)
        burst_frequency_mean = np.mean(burst_frequency)  # <fb>
        duty_cycle_mean = np.mean(duty_cycle)  # <dc>
        num_sw = np.size(self.convert_trace_to_spike_times_downward(voltage, times, threshold['threshold_slow_wave']))

        # num_spikes = np.size(self.convert_trace_to_spike_times_upward(voltage, times, threshold['threshold_spike']))
        num_spike_per_burst = self.clean_up_num_spikes_per_burst(
            np.diff(np.insert(num_bursts['be_loc'], 0, 0)))
        # num_mid = np.array([np.size(times_before_th_downward), np.size(times_before_th_downward2)])
        nspike_mid1 = self.clean_up_num_spikes_per_burst(
            self.calculate_num_spikes_per_burst(times_before_th_downward, temporal_interval))
        nspike_mid2 = self.clean_up_num_spikes_per_burst(
            self.calculate_num_spikes_per_burst(times_before_th_downward2, temporal_interval))
        spike_per_burst_mid = (nspike_mid1, nspike_mid2)

        return {'burst_frequency_mean': burst_frequency_mean,
                'duty_cycle_mean': duty_cycle_mean,
                'times_before_th_downward': times_before_th_downward,
                'times_before_th_downward2': times_before_th_downward2,
                'num_sw': num_sw,
                'e_lag': e_lag,
                # 'num_spikes': num_spikes,
                # 'num_mid': num_mid,
                'burst_frequency': burst_frequency,
                'duty_cycle': duty_cycle,
                'spike_per_burst': num_spike_per_burst,
                'spike_per_burst_mid': spike_per_burst_mid,
                }
