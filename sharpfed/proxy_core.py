# -*- coding: utf-8 -*-
import os
import time
import pickle
import shutil
import numpy as np
import tensorflow as tf
from pathlib import Path

from sharpfed.functions import aggregate_model_updates, \
                               log_print, \
                               download_file_from_remote, \
                               upload_file_to_remote, \
                               create_folder_on_remote, \
                               create_file_on_remote, \
                               remove_folder_on_remote, \
                               remove_file_on_remote, \
                               get_filenames_in_remote_folder
SLEEPING_TIME = 5

class RemoteModeClient:
    def __init__(self):
        self.__client_id = -1
        self.__server_to_connect_id = -1
        self.__client_cache_folder_local = -1
        self.__comm_channel_folder_proxy = -1

        self.__proxy_addr = -1
        self.__proxy_port = -1
        self.__proxy_username = -1
        self.__proxy_passwd = -1

    def initialize(self,
                   client_id: int,
                   server_to_connect_id: int,
                   meta_data: dict,
                   client_cache_folder_local: str,
                   comm_channel_folder_proxy: str,
                   proxy_addr: str,
                   proxy_port: int,
                   proxy_username: str,
                   proxy_passwd: str) -> bool:
    
        self.__client_id = client_id
        self.__server_to_connect_id = server_to_connect_id
    
        self.__client_cache_folder_local = Path(client_cache_folder_local)
        self.__comm_channel_folder_proxy = Path(comm_channel_folder_proxy)
        self.__client_cache_folder_proxy = Path(self.__comm_channel_folder_proxy).joinpath(str(self.__client_id))

        self.__proxy_addr = proxy_addr
        self.__proxy_port = proxy_port
        self.__proxy_username = proxy_username
        self.__proxy_passwd = proxy_passwd

        # --------------------- Refresh local client cache folder -------------------- #
        if os.path.exists(self.__client_cache_folder_local):
            shutil.rmtree(self.__client_cache_folder_local)
            os.mkdir(self.__client_cache_folder_local)
        else:
            os.mkdir(self.__client_cache_folder_local)
        # ---------------------------------------------------------------------------- #

        # ---------------- Refresh client cache folder on proxy server --------------- #
        remove_folder_on_remote(remote_folder_path=str(self.__client_cache_folder_proxy),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__client_cache_folder_proxy),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        # ---------------------------------------------------------------------------- #

        # -------------------------- Save client's meta data ------------------------- #
        file = open(self.__client_cache_folder_local.joinpath("meta_data.pkl"), "wb")
        pickle.dump(meta_data, file)
        file.close()

        upload_file_to_remote(remote_file_path=str(self.__client_cache_folder_proxy.joinpath("meta_data.pkl")),
                              local_file_path=str(self.__client_cache_folder_local.joinpath("meta_data.pkl")),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        # ---------------------------------------------------------------------------- #

        #------------ Print initialization info ------------#
        log_print('+-------------------------- Client Initialization Finished --------------------------+', color='m')
        log_print('+---> Client ID: %d' % self.__client_id, color='m')
        log_print('+---> Server to connect ID: %d' % self.__server_to_connect_id, color='m')
        log_print('+---> Proxy Address: %s' % self.__proxy_addr, color='m')
        log_print('+------------------------------------------------------------------------------------+', color='m')
        #---------------------------------------------------#
        return True

    def get_client_id(self) -> int:
        return self.__client_id

    def get_server_to_connect_id(self) -> int:
        return self.__server_to_connect_id

    def __get_round_num(self) -> int:
        round_num = get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RoundNum")),
                                                   remote_addr=self.__proxy_addr,
                                                   remote_port=self.__proxy_port,
                                                   remote_username=self.__proxy_username,
                                                   remote_passwd=self.__proxy_passwd)
        return int(round_num[0])

    def __request_connect_to_server(self) -> bool:
        create_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RequestToConnect", str(self.__client_id))),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        return True

    def __is_connected_to_server(self) -> bool:
        connected_clients = get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("Connected")),
                                                           remote_addr=self.__proxy_addr,
                                                           remote_port=self.__proxy_port,
                                                           remote_username=self.__proxy_username,
                                                           remote_passwd=self.__proxy_passwd)
        if str(self.__client_id) in connected_clients:
            return True
        else:
            return False

    def __is_selected(self) -> bool:
        round_selected_clients = get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RoundSelected")),
                                                                remote_addr=self.__proxy_addr,
                                                                remote_port=self.__proxy_port,
                                                                remote_username=self.__proxy_username,
                                                                remote_passwd=self.__proxy_passwd)
        if str(self.__client_id) in round_selected_clients:
            return True
        else:
            return False

    def local_update(self, global_model_parameters: list) -> tuple:
        layermask_vector = np.ones_like(len(global_model_parameters))
        local_model_updates = [np.ones_like(global_model_parameters[layer_idx]*layermask_vector[layer_idx]) for layer_idx in range(len(global_model_parameters))]
        return local_model_updates, layermask_vector
    
    def __update(self) -> bool:
        r = self.__get_round_num()
        #------------------- Load global model parameters -------------------#
        download_file_from_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundModels", "global_model_parameters_%d.pkl" % r)),
                                  local_file_path=str(self.__client_cache_folder_local.joinpath("global_model_parameters_%d.pkl" % r)),
                                  remote_addr=self.__proxy_addr,
                                  remote_port=self.__proxy_port,
                                  remote_username=self.__proxy_username,
                                  remote_passwd=self.__proxy_passwd)
        file = open(self.__client_cache_folder_local.joinpath("global_model_parameters_%d.pkl" % r), "rb")
        last_round_global_model_parameters = pickle.load(file)
        file.close()
        #--------------------------------------------------------------------#

        # ---------------------- Local train and upload updates ---------------------- #
        local_model_updates, layermask_vector = self.local_update(last_round_global_model_parameters)

        file = open(self.__client_cache_folder_local.joinpath("model_updates.pkl"), "wb")
        pickle.dump(local_model_updates, file)
        file.close()

        file = open(self.__client_cache_folder_local.joinpath("layermask.pkl"), "wb")
        pickle.dump(layermask_vector, file)
        file.close()

        upload_file_to_remote(remote_file_path=str(self.__client_cache_folder_proxy.joinpath("model_updates.pkl")),
                              local_file_path=str(self.__client_cache_folder_local.joinpath("model_updates.pkl")),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        upload_file_to_remote(remote_file_path=str(self.__client_cache_folder_proxy.joinpath("layermask.pkl")),
                              local_file_path=str(self.__client_cache_folder_local.joinpath("layermask.pkl")),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        remove_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundSelected", str(self.__client_id))),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        # ---------------------------------------------------------------------------- #
        return True

    def start(self) -> bool:
        # -------------------------- Waiting to be connected ------------------------- #
        while self.__is_connected_to_server()==False:
            log_print("[Client %d Info] Server not connected..." % self.__client_id, color='m')
            self.__request_connect_to_server()
            time.sleep(SLEEPING_TIME*10)
        log_print("[Client %d Info] Server successfully connected!" % self.__client_id, color='g')
        # ---------------------------------------------------------------------------- #

        # ------------------- Waiting to be selected & local update ------------------ #
        while 'exit' not in get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy),
                                                           remote_addr=self.__proxy_addr,
                                                           remote_port=self.__proxy_port,
                                                           remote_username=self.__proxy_username,
                                                           remote_passwd=self.__proxy_passwd):
            if self.__is_selected():
                log_print("[Client %d Info] Be selected to participate in the update and begin local update..." % self.__client_id, color='m')
                self.__update()
                log_print("[Client %d Info] Local update finished!" % self.__client_id, color='g')
            else:
                time.sleep(SLEEPING_TIME*10)
        # ---------------------------------------------------------------------------- #

        log_print("[Client %d Info] Training finished!" % self.__client_id, color='g')
        return True

class RemoteModeServer:
    def __init__(self):
        self.__num_rounds = -1
        self.__models_save_folder_local = -1
        self.__min_connected_clients_to_start = -1
        self.__comm_channel_folder_proxy = -1
        self.__server_id = -1
        self.__server_cache_folder_local = -1

        self.__proxy_addr = -1
        self.__proxy_port = -1
        self.__proxy_username = -1
        self.__proxy_passwd = -1

    def initialize(self,
                   server_id: int,
                   server_cache_folder_local: str,
                   initial_model: tf.keras.Model,
                   num_rounds: int,
                   min_connected_clients_to_start: int,
                   comm_channel_folder_proxy: str,
                   proxy_addr: str,
                   proxy_port: int,
                   proxy_username: str,
                   proxy_passwd: str) -> bool:

        self.__server_id = server_id
        self.__server_cache_folder_local = Path(server_cache_folder_local)
        self.__models_save_folder_local = self.__server_cache_folder_local.joinpath("RoundModels")

        self.__num_rounds = num_rounds
        self.__min_connected_clients_to_start = min_connected_clients_to_start

        assert self.__num_rounds>0
        assert self.__min_connected_clients_to_start>=2

        self.__comm_channel_folder_proxy = Path(comm_channel_folder_proxy)
        self.__proxy_addr = proxy_addr
        self.__proxy_port = proxy_port
        self.__proxy_username = proxy_username
        self.__proxy_passwd = proxy_passwd

        # --------------------- Refresh local server cache folder -------------------- #
        if os.path.exists(self.__server_cache_folder_local):
            shutil.rmtree(self.__server_cache_folder_local)
            os.mkdir(self.__server_cache_folder_local)
        else:
            os.mkdir(self.__server_cache_folder_local)
        os.mkdir(self.__server_cache_folder_local.joinpath(".CacheFiles"))
        os.mkdir(self.__models_save_folder_local)
        # ---------------------------------------------------------------------------- #

        # ----------- Refresh communication channel folder on proxy server ----------- #
        remove_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RequestToConnect")),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("Connected")),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RoundNum")),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RoundSelected")),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        create_folder_on_remote(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RoundModels")),
                                remote_addr=self.__proxy_addr,
                                remote_port=self.__proxy_port,
                                remote_username=self.__proxy_username,
                                remote_passwd=self.__proxy_passwd)
        # ---------------------------------------------------------------------------- #

        # ------------------ Initialize the global model parameters ------------------ #
        initial_model_parameters = initial_model.get_weights()
        file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_0.pkl"), "wb")
        pickle.dump(initial_model_parameters, file)
        file.close()
        upload_file_to_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundModels", "global_model_parameters_0.pkl")),
                              local_file_path=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_0.pkl")),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        # ---------------------------------------------------------------------------- #

        #---------------------------- Print initialization info ----------------------------#
        log_print('+-------------------------- Server Initialization Finished --------------------------+', color='m')
        log_print('+---> Server ID: %d' % self.__server_id, color='m')
        log_print('+---> FL iteration round: %d' % self.__num_rounds, color='m')
        log_print('+---> Min connected clients to start: %d' % self.__min_connected_clients_to_start, color='m')
        log_print('+---> Round models save folder: %s' % str(self.__models_save_folder_local), color='m')
        log_print('+---> Proxy server address: %s' % self.__proxy_addr, color='m')
        log_print('+------------------------------------------------------------------------------------+', color='m')
        #-----------------------------------------------------------------------------------#
        return True

    def __connect_clients(self, client_id_list: list) -> bool:
        for client_id in client_id_list:
            remove_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RequestToConnect", str(client_id))),
                                  remote_addr=self.__proxy_addr,
                                  remote_port=self.__proxy_port,
                                  remote_username=self.__proxy_username,
                                  remote_passwd=self.__proxy_passwd)
            create_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("Connected", str(client_id))),
                                  remote_addr=self.__proxy_addr,
                                  remote_port=self.__proxy_port,
                                  remote_username=self.__proxy_username,
                                  remote_passwd=self.__proxy_passwd)
        return True

    def __get_request_to_connect_clients_id(self) -> list:
        request_to_connect_clients_id = get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RequestToConnect")),
                                                                       remote_addr=self.__proxy_addr,
                                                                       remote_port=self.__proxy_port,
                                                                       remote_username=self.__proxy_username,
                                                                       remote_passwd=self.__proxy_passwd)
        return [int(i) for i in request_to_connect_clients_id]

    def __get_connected_clients_id(self) -> list:
        connected_clients_id = get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("Connected")),
                                                              remote_addr=self.__proxy_addr,
                                                              remote_port=self.__proxy_port,
                                                              remote_username=self.__proxy_username,
                                                              remote_passwd=self.__proxy_passwd)
        return [int(i) for i in connected_clients_id]

    def __get_round_selected_clients_id(self) -> list:
        round_selected_clients_id = get_filenames_in_remote_folder(remote_folder_path=str(self.__comm_channel_folder_proxy.joinpath("RoundSelected")),
                                                                   remote_addr=self.__proxy_addr,
                                                                   remote_port=self.__proxy_port,
                                                                   remote_username=self.__proxy_username,
                                                                   remote_passwd=self.__proxy_passwd)
        return [int(i) for i in round_selected_clients_id]

    def get_server_id(self) -> int:
        return self.__server_id

    def round_response_to_client_connection_request(self, clients_meta_data: dict) -> list:
        '''Determine whether to accept the connection request from the client.

        Input:
            `clients_meta_data`: dict, the meta data of clients.
            Each key is the client id, the corresponding value is a dict containing the meta data that you defined,
            for example, `{1:{"num_training_instances":300}, 2:{"num_training_instances":200}}`.
        
        Output:
            `request_to_connect_clients_id`: list of int, the client id of the clients whose connection request is accepted.
        '''
        request_to_connect_clients_id = []

        # ----------------------- Your selection strategy here. ---------------------- #
        for client_id in clients_meta_data.keys():
            request_to_connect_clients_id.append(client_id)
        # ---------------------------------------------------------------------------- #

        return request_to_connect_clients_id

    def round_client_selection(self, clients_meta_data: dict) -> list:
        '''Client selection for each round.
        
        Input: 
            `clients_meta_data`: dict, the meta data of clients. 
            Each key is the client id, the corresponding value is a dict containing the meta data that you defined,
            for example, `{1:{"num_training_instances":300}, 2:{"num_training_instances":200}}`.
            
        Output:
            `selected_clients_id`: list of int, the client id of the clients selected for this round.
        '''
        selected_clients_id = []

        # ----------------------- Your selection strategy here. ---------------------- #
        for client_id in clients_meta_data.keys():
            selected_clients_id.append(client_id)
        # ---------------------------------------------------------------------------- #

        return selected_clients_id

    def set_client_aggregation_weight(self, clients_meta_data: dict) -> dict:
        '''Set the aggregation weight for each client.

        Input: 
            `clients_meta_data`: dict, the meta data of clients. 
            Each key is the client id, the corresponding value is a dict containing the meta data that you defined,
            for example, `{1:{"num_training_instances":300}, 2:{"num_training_instances":200}}`.

        Output:
            `client_aggregation_weight`: dict, the aggregation weight for clients. 
            Each key is the client id, the corresponding value is a float which is the aggregation weight for this client.
            Note that the aggregation weights DOES NOT need to be normalized.
        '''

        client_aggregation_weight = dict()

        # ------------------ Your aggregation weight strategy here. ------------------ #
        for client_id in clients_meta_data.keys():
            client_aggregation_weight[client_id] = 1.0
        # ---------------------------------------------------------------------------- #

        return client_aggregation_weight

    def start(self) -> bool:
        for r in range(self.__num_rounds):
            remove_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundNum", str(r-1))),
                                  remote_addr=self.__proxy_addr,
                                  remote_port=self.__proxy_port,
                                  remote_username=self.__proxy_username,
                                  remote_passwd=self.__proxy_passwd)
            create_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundNum", str(r))),
                                  remote_addr=self.__proxy_addr,
                                  remote_port=self.__proxy_port,
                                  remote_username=self.__proxy_username,
                                  remote_passwd=self.__proxy_passwd)

            # ---------------------- Waiting enough clients to start --------------------- #
            while True:
                request_to_connect_clients_id = self.__get_request_to_connect_clients_id()
                log_print("[Server Info] Round %d, %d clients request to connect" % (r, len(request_to_connect_clients_id)), color='g')
                client_meta_data = dict()
                for client_id in request_to_connect_clients_id:
                    download_file_from_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath(str(client_id), "meta_data.pkl")),
                                              local_file_path=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "meta_data_%d.pkl" % client_id)),
                                              remote_addr=self.__proxy_addr,
                                              remote_port=self.__proxy_port,
                                              remote_username=self.__proxy_username,
                                              remote_passwd=self.__proxy_passwd)
                    file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "meta_data_%d.pkl" % client_id), "rb")
                    this_client_meta_data = pickle.load(file)
                    file.close()
                    client_meta_data[client_id] = this_client_meta_data
                self.__connect_clients(self.round_response_to_client_connection_request(client_meta_data))
                time.sleep(SLEEPING_TIME)

                connected_clients_id = self.__get_connected_clients_id()
                if len(connected_clients_id)<self.__min_connected_clients_to_start:
                    log_print("[Server Info] Round %d, current %d clients connected, min %d, waiting more clients to participate..."
                              % (r, len(connected_clients_id), self.__min_connected_clients_to_start), color='m')
                    time.sleep(SLEEPING_TIME)
                else:
                    break
            time.sleep(SLEEPING_TIME)
            # ---------------------------------------------------------------------------- #

            # ------------------------------ Round selection ----------------------------- #
            client_meta_data = dict()
            for client_id in connected_clients_id:
                file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "meta_data_%d.pkl" % client_id), "rb")
                this_client_meta_data = pickle.load(file)
                file.close()
                client_meta_data[client_id] = this_client_meta_data

            round_selected_clients_id = self.round_client_selection(client_meta_data)
            for client_id in round_selected_clients_id:
                create_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundSelected", str(client_id))),
                                      remote_addr=self.__proxy_addr,
                                      remote_port=self.__proxy_port,
                                      remote_username=self.__proxy_username,
                                      remote_passwd=self.__proxy_passwd)
            log_print("[Server Info] Round %d, %d/%d clients are selected to participate"
                      % (r, len(round_selected_clients_id), len(connected_clients_id)), color='g')
            time.sleep(SLEEPING_TIME)
            # ---------------------------------------------------------------------------- #

            # ------------ Waiting for selected clients to finish local update ----------- #
            while True:
                remaining_selected_clients = self.__get_round_selected_clients_id()
                if len(remaining_selected_clients)==0:
                    break

                log_print("[Server Info] Round %d, waiting %d/%d selected clients to finish local update..."
                          % (r, len(remaining_selected_clients), len(round_selected_clients_id)), color='m')
                time.sleep(SLEEPING_TIME)
            # ---------------------------------------------------------------------------- #

            # --------- Aggregate model updates and distribute aggregated updates -------- #
            client_model_updates = dict()
            client_meta_data = dict()
            client_layermask = dict()
            for client_id in round_selected_clients_id:
                download_file_from_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath(str(client_id), "model_updates.pkl")),
                                          local_file_path=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "model_updates_%d.pkl" % client_id)),
                                          remote_addr=self.__proxy_addr,
                                          remote_port=self.__proxy_port,
                                          remote_username=self.__proxy_username,
                                          remote_passwd=self.__proxy_passwd)
                file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "model_updates_%d.pkl" % client_id), "rb")
                this_client_model_updates = pickle.load(file)
                file.close()
                client_model_updates[client_id] = this_client_model_updates

                download_file_from_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath(str(client_id), "layermask.pkl")),
                                          local_file_path=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "layermask_%d.pkl" % client_id)),
                                          remote_addr=self.__proxy_addr,
                                          remote_port=self.__proxy_port,
                                          remote_username=self.__proxy_username,
                                          remote_passwd=self.__proxy_passwd)
                file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "layermask_%d.pkl" % client_id), "rb")
                this_client_layermask = pickle.load(file)
                file.close()
                client_layermask[client_id] = this_client_layermask

                file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "meta_data_%d.pkl" % client_id), "rb")
                this_client_meta_data = pickle.load(file)
                file.close()
                client_meta_data[client_id] = this_client_meta_data

            aggregated_model_updates = aggregate_model_updates(client_model_updates=client_model_updates,
                                                               client_aggregation_weight=self.set_client_aggregation_weight(client_meta_data),
                                                               client_layermask=client_layermask)
            # Update the global model parameters
            download_file_from_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundModels", "global_model_parameters_%d.pkl" % r)),
                                      local_file_path=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % r)),
                                      remote_addr=self.__proxy_addr,
                                      remote_port=self.__proxy_port,
                                      remote_username=self.__proxy_username,
                                      remote_passwd=self.__proxy_passwd)
            file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % r), "rb")
            last_round_global_model_parameters = pickle.load(file)
            file.close()

            updated_global_model_parameters = [last_round_global_model_parameters[layer_idx] + aggregated_model_updates[layer_idx] 
                                                for layer_idx in range(len(last_round_global_model_parameters))]
            
            file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % (r+1)), "wb")
            pickle.dump(updated_global_model_parameters, file)
            file.close()

            upload_file_to_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath("RoundModels", "global_model_parameters_%d.pkl" % (r+1))),
                                  local_file_path=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % (r+1))),
                                  remote_addr=self.__proxy_addr,
                                  remote_port=self.__proxy_port,
                                  remote_username=self.__proxy_username,
                                  remote_passwd=self.__proxy_passwd)
            time.sleep(SLEEPING_TIME)

            # Save the updated global model parameters
            file = open(self.__models_save_folder_local.joinpath("global_model_parameters_%d.pkl" % (r+1)), "wb")
            pickle.dump(updated_global_model_parameters, file)
            file.close()
            log_print("[Server Info] Round %d finished! " % r, 'g')
            time.sleep(SLEEPING_TIME)
            # ---------------------------------------------------------------------------- #

        # ----------------------------- Training finished ---------------------------- #
        create_file_on_remote(remote_file_path=str(self.__comm_channel_folder_proxy.joinpath('exit')),
                              remote_addr=self.__proxy_addr,
                              remote_port=self.__proxy_port,
                              remote_username=self.__proxy_username,
                              remote_passwd=self.__proxy_passwd)
        log_print("[Server Info] Training finished!", 'b')
        # ---------------------------------------------------------------------------- #
        return True