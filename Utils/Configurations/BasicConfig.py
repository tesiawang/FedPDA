# -*- coding: utf-8 -*-
import sionna
import numpy as np

class BasicConfig():
    def __init__(self,
                 tdl_model: str = 'A',
                 delay_spread: float = 30e-9,
                 min_speed: float = 0.0,
                 max_speed: float = 0.0):
        super().__init__()

        # Provided parameters
        self._tdl_model = tdl_model
        self._delay_spread = delay_spread
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._cyclic_prefix_length = 16
        self._pilot_ofdm_symbol_indices = [2,11]

        # System parameters
        self._num_ut_ant = 1
        self._num_bs_ant = 1
        self._carrier_frequency = 4e9
        self._subcarrier_spacing = 15e3
        self._fft_size = 264
        self._num_ofdm_symbols = 14
        self._num_streams_per_tx = 1
        self._dc_null = False
        self._num_guard_carriers = [0,0]
        self._pilot_pattern = "kronecker"
        self._num_bits_per_symbol = 4
        self._coderate = 658/1024

        # Required system components
        self._sm = sionna.mimo.StreamManagement(np.array([[1]]),
                                                self._num_streams_per_tx)
        self._rg = sionna.ofdm.ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                            fft_size=self._fft_size,
                                            subcarrier_spacing = self._subcarrier_spacing,
                                            num_tx=1,
                                            num_streams_per_tx=self._num_streams_per_tx,
                                            cyclic_prefix_length=self._cyclic_prefix_length,
                                            num_guard_carriers=self._num_guard_carriers,
                                            dc_null=self._dc_null,
                                            pilot_pattern=self._pilot_pattern,
                                            pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n*self._coderate)

        self._ut_array = sionna.channel.tr38901.Antenna(polarization="single",
                                                        polarization_type="V",
                                                        antenna_pattern="38.901",
                                                        carrier_frequency=self._carrier_frequency)

        self._bs_array = sionna.channel.tr38901.Antenna(polarization="single",
                                                        polarization_type="V",
                                                        antenna_pattern="38.901",
                                                        carrier_frequency=self._carrier_frequency)

        self._tdl = sionna.channel.tr38901.TDL(model=self._tdl_model,
                                               delay_spread=self._delay_spread,
                                               carrier_frequency=self._carrier_frequency,
                                               min_speed=self._min_speed,
                                               max_speed=self._max_speed,
                                               num_rx_ant=1,
                                               num_tx_ant=1)

        self._frequencies = sionna.channel.subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)
        self._channel_freq = sionna.channel.ApplyOFDMChannel(add_awgn=True)

        self._binary_source = sionna.utils.BinarySource()
        self._encoder = sionna.fec.ldpc.encoding.LDPC5GEncoder(self._k, self._n)
        self._mapper = sionna.mapping.Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = sionna.ofdm.ResourceGridMapper(self._rg)
        self._rg_demapper = sionna.ofdm.ResourceGridDemapper(self._rg, self._sm)

        self._zf_precoder = sionna.ofdm.ZFPrecoder(self._rg,
                                                   self._sm,
                                                   return_effective_channel=True)

        self._ls_est = sionna.ofdm.LSChannelEstimator(self._rg, interpolation_type="lin")
        self._lmmse_equ = sionna.ofdm.LMMSEEqualizer(self._rg, self._sm)
        self._demapper = sionna.mapping.Demapper("app", "qam", self._num_bits_per_symbol)
        self._decoder = sionna.fec.ldpc.decoding.LDPC5GDecoder(self._encoder, hard_out=True)
        self._remove_nulled_scs = sionna.ofdm.RemoveNulledSubcarriers(self._rg)