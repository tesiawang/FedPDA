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
                               create_file_on_local, \
                               remove_file_on_local, \
                               get_filenames_in_local_folder
SLEEPING_TIME = 5

class LocalModeClient:
    def __init__(self,
                 client_id: int,
                 server_to_connect_id: int,
                 meta_data: dict,
                 client_cache_folder_local: str,
                 comm_channel_folder: str) -> bool:
    
        self.__client_id = client_id
        self.__server_to_connect_id = server_to_connect_id
        self.__client_cache_folder_local = Path(client_cache_folder_local)
        self.__comm_channel_folder = Path(comm_channel_folder)

        # --------------------- Refresh local client cache folder -------------------- #
        if os.path.exists(self.__client_cache_folder_local):
            shutil.rmtree(self.__client_cache_folder_local)
            os.mkdir(self.__client_cache_folder_local)
        else:
            os.mkdir(self.__client_cache_folder_local)
        # ---------------------------------------------------------------------------- #

        # ---------------- Refresh client cache folder on comm channel --------------- #
        if os.path.exists(self.__comm_channel_folder.joinpath(str(self.__client_id))):
            shutil.rmtree(self.__comm_channel_folder.joinpath(str(self.__client_id)))
            os.mkdir(self.__comm_channel_folder.joinpath(str(self.__client_id)))
        else:
            os.mkdir(self.__comm_channel_folder.joinpath(str(self.__client_id)))
        # ---------------------------------------------------------------------------- #

        # -------------------------- Save client's meta data ------------------------- #
        file = open(self.__client_cache_folder_local.joinpath("meta_data.pkl"), "wb")
        pickle.dump(meta_data, file)
        file.close()

        shutil.copyfile(src=str(self.__client_cache_folder_local.joinpath("meta_data.pkl")),
                        dst=str(self.__comm_channel_folder.joinpath(str(self.__client_id), "meta_data.pkl")))
        # ---------------------------------------------------------------------------- #

        #------------ Print initialization info ------------#
        log_print('Client Initialization Finished', color='m')
        log_print('---> Client ID: %d' % self.__client_id, color='m')
        log_print('---> Server to connect ID: %d' % self.__server_to_connect_id, color='m')
        #---------------------------------------------------#

    def get_client_id(self) -> int:
        return self.__client_id

    def get_server_to_connect_id(self) -> int:
        return self.__server_to_connect_id

    def get_round_num(self) -> int:
        round_num = get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder.joinpath("RoundNum")))
        return int(round_num[0])

    def __request_connect_to_server(self) -> bool:
        create_file_on_local(path=str(self.__comm_channel_folder.joinpath("RequestToConnect", str(self.__client_id))))
        return True

    def __is_connected_to_server(self) -> bool:
        connected_clients = get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder.joinpath("Connected")))
        if str(self.__client_id) in connected_clients:
            return True
        else:
            return False

    def __is_selected(self) -> bool:
        round_selected_clients = get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder.joinpath("RoundSelected")))
        if str(self.__client_id) in round_selected_clients:
            return True
        else:
            return False

    def local_update(self, global_model_parameters: list) -> tuple:
        layermask_vector = np.ones_like(len(global_model_parameters))
        local_model_updates = [np.ones_like(global_model_parameters[layer_idx]*layermask_vector[layer_idx]) for layer_idx in range(len(global_model_parameters))]
        return local_model_updates, layermask_vector

    def __update(self) -> bool:
        r = self.get_round_num()
        # ----------------------- Load global model parameters ----------------------- #
        shutil.copyfile(src=str(self.__comm_channel_folder.joinpath("RoundModels", "global_model_parameters_%d.pkl" % r)),
                        dst=str(self.__client_cache_folder_local.joinpath("global_model_parameters_%d.pkl" % r)))
        file = open(self.__client_cache_folder_local.joinpath("global_model_parameters_%d.pkl" % r), "rb")
        last_round_global_model_parameters = pickle.load(file)
        file.close()
        # ---------------------------------------------------------------------------- #

        # ---------------------- Local train and upload updates ---------------------- #
        local_model_updates, layermask_vector = self.local_update(last_round_global_model_parameters)

        file = open(self.__client_cache_folder_local.joinpath("model_updates.pkl"), "wb")
        pickle.dump(local_model_updates, file)
        file.close()

        file = open(self.__client_cache_folder_local.joinpath("layermask.pkl"), "wb")
        pickle.dump(layermask_vector, file)
        file.close()

        shutil.copyfile(src=str(self.__client_cache_folder_local.joinpath("model_updates.pkl")),
                        dst=str(self.__comm_channel_folder.joinpath(str(self.__client_id), "model_updates.pkl")))

        shutil.copyfile(src=str(self.__client_cache_folder_local.joinpath("layermask.pkl")),
                        dst=str(self.__comm_channel_folder.joinpath(str(self.__client_id), "layermask.pkl")))

        remove_file_on_local(path=str(self.__comm_channel_folder.joinpath("RoundSelected", str(self.__client_id))))
        # ---------------------------------------------------------------------------- #
        return True

    def start(self) -> bool:
        # -------------------------- Waiting to be connected ------------------------- #
        while self.__is_connected_to_server()==False:
            log_print("[Client %d Info] Server not connected..." % self.__client_id, color='m')
            self.__request_connect_to_server()
            time.sleep(SLEEPING_TIME)
        log_print("[Client %d Info] Server successfully connected!" % self.__client_id, color='g')
        # ---------------------------------------------------------------------------- #

        # ------------------- Waiting to be selected & local update ------------------ #
        while 'exit' not in get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder)):
            if self.__is_selected():
                log_print("[Client %d Info] Be selected to participate in the update and begin local update..." % self.__client_id, color='m')
                self.__update()
                log_print("[Client %d Info] Local update finished!" % self.__client_id, color='g')
            else:
                time.sleep(SLEEPING_TIME)
        # ---------------------------------------------------------------------------- #

        log_print("[Client %d Info] Training finished!" % self.__client_id, color='g')
        return True

class LocalModeServer:
    def __init__(self,
                 server_id: int,
                 server_cache_folder_local: str,
                 initial_model: tf.keras.Model,
                 num_rounds: int,
                 min_connected_clients_to_start: int,
                 comm_channel_folder: str) -> bool:

        self.__server_id = server_id
        self.__server_cache_folder_local = Path(server_cache_folder_local)
        self.__models_save_folder_local = self.__server_cache_folder_local.joinpath("RoundModels")

        self.__num_rounds = num_rounds
        self.__min_connected_clients_to_start = min_connected_clients_to_start

        assert self.__num_rounds>0
        assert self.__min_connected_clients_to_start>=2

        self.__comm_channel_folder = Path(comm_channel_folder)

        # --------------------- Refresh local server cache folder -------------------- #
        if os.path.exists(self.__server_cache_folder_local):
            shutil.rmtree(self.__server_cache_folder_local)
            os.mkdir(self.__server_cache_folder_local)
        else:
            os.mkdir(self.__server_cache_folder_local)
        os.mkdir(self.__server_cache_folder_local.joinpath(".CacheFiles"))
        os.mkdir(self.__models_save_folder_local)
        # ---------------------------------------------------------------------------- #

        # ------------------- Refresh communication channel folder ------------------- #
        if os.path.exists(self.__comm_channel_folder):
            shutil.rmtree(self.__comm_channel_folder)
            os.mkdir(self.__comm_channel_folder)
        else:
            os.mkdir(self.__comm_channel_folder)
        os.mkdir(str(self.__comm_channel_folder.joinpath("RequestToConnect")))
        os.mkdir(str(self.__comm_channel_folder.joinpath("Connected")))
        os.mkdir(str(self.__comm_channel_folder.joinpath("RoundNum")))
        os.mkdir(str(self.__comm_channel_folder.joinpath("RoundSelected")))
        os.mkdir(str(self.__comm_channel_folder.joinpath("RoundModels")))
        # ---------------------------------------------------------------------------- #

        # ------------------ Initialize the global model parameters ------------------ #
        initial_model_parameters = initial_model.get_weights()
        file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_0.pkl"), "wb")
        pickle.dump(initial_model_parameters, file)
        file.close()

        shutil.copyfile(src=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_0.pkl")),
                        dst=str(self.__models_save_folder_local.joinpath("global_model_parameters_0.pkl")))

        shutil.copyfile(src=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_0.pkl")),
                        dst=str(self.__comm_channel_folder.joinpath("RoundModels", "global_model_parameters_0.pkl")))
        # ---------------------------------------------------------------------------- #

        #---------------------------- Print initialization info ----------------------------#
        log_print('Server Initialization Finished', color='m')
        log_print('---> Server ID: %d' % self.__server_id, color='m')
        log_print('---> FL iteration round: %d' % self.__num_rounds, color='m')
        log_print('---> Min connected clients to start: %d' % self.__min_connected_clients_to_start, color='m')
        log_print('---> Round models save folder: %s' % str(self.__models_save_folder_local), color='m')
        #-----------------------------------------------------------------------------------#

    def __connect_clients(self, client_id_list: list) -> bool:
        for client_id in client_id_list:
            remove_file_on_local(path=str(self.__comm_channel_folder.joinpath("RequestToConnect", str(client_id))))
            create_file_on_local(path=str(self.__comm_channel_folder.joinpath("Connected", str(client_id))))
        return True

    def __get_request_to_connect_clients_id(self) -> list:
        request_to_connect_clients_id = get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder.joinpath("RequestToConnect")))
        return [int(i) for i in request_to_connect_clients_id]

    def __get_connected_clients_id(self) -> list:
        connected_clients_id = get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder.joinpath("Connected")))
        return [int(i) for i in connected_clients_id]

    def __get_round_selected_clients_id(self) -> list:
        round_selected_clients_id = get_filenames_in_local_folder(local_folder_path=str(self.__comm_channel_folder.joinpath("RoundSelected")))
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
            remove_file_on_local(path=str(self.__comm_channel_folder.joinpath("RoundNum", str(r-1))))
            create_file_on_local(path=str(self.__comm_channel_folder.joinpath("RoundNum", str(r))))

            # ---------------------- Waiting enough clients to start --------------------- #
            while True:
                request_to_connect_clients_id = self.__get_request_to_connect_clients_id()
                log_print("[Server Info] Round %d, %d clients request to connect" % (r, len(request_to_connect_clients_id)), color='m')

                client_meta_data = dict()
                for client_id in request_to_connect_clients_id:
                    shutil.copy(src=str(self.__comm_channel_folder.joinpath(str(client_id), "meta_data.pkl")),
                                dst=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "meta_data_%d.pkl" % client_id)))

                    file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "meta_data_%d.pkl" % client_id), "rb")
                    this_client_meta_data = pickle.load(file)
                    file.close()
                    client_meta_data[client_id] = this_client_meta_data
                self.__connect_clients(self.round_response_to_client_connection_request(client_meta_data))

                connected_clients_id = self.__get_connected_clients_id()
                if len(connected_clients_id)<self.__min_connected_clients_to_start:
                    log_print("[Server Info] Round %d, current %d clients connected, min %d, waiting more clients to participate..."
                              % (r, len(connected_clients_id), self.__min_connected_clients_to_start), color='m')
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
                create_file_on_local(path=str(self.__comm_channel_folder.joinpath("RoundSelected", str(client_id))))
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
                time.sleep(SLEEPING_TIME*10)
            # ---------------------------------------------------------------------------- #

            # --------- Aggregate model updates and distribute aggregated updates -------- #
            client_model_updates = dict()
            client_layermask = dict()
            client_meta_data = dict()
            for client_id in round_selected_clients_id:
                shutil.copy(src=str(self.__comm_channel_folder.joinpath(str(client_id), "model_updates.pkl")),
                            dst=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "model_updates_%d.pkl" % client_id)))
                file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "model_updates_%d.pkl" % client_id), "rb")
                this_client_model_updates = pickle.load(file)
                file.close()
                client_model_updates[client_id] = this_client_model_updates

                shutil.copy(src=str(self.__comm_channel_folder.joinpath(str(client_id), "layermask.pkl")),
                            dst=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "layermask_%d.pkl" % client_id)))
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
            shutil.copy(src=str(self.__comm_channel_folder.joinpath("RoundModels", "global_model_parameters_%d.pkl" % r)),
                        dst=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % r)))
            file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % r), "rb")
            last_round_global_model_parameters = pickle.load(file)
            file.close()

            updated_global_model_parameters = [last_round_global_model_parameters[layer_idx] + aggregated_model_updates[layer_idx] 
                                               for layer_idx in range(len(last_round_global_model_parameters))]
            
            file = open(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % (r+1)), "wb")
            pickle.dump(updated_global_model_parameters, file)
            file.close()
            shutil.copy(src=str(self.__server_cache_folder_local.joinpath(".CacheFiles", "global_model_parameters_%d.pkl" % (r+1))),
                        dst=str(self.__comm_channel_folder.joinpath("RoundModels", "global_model_parameters_%d.pkl" % (r+1))))

            # Save the updated global model parameters
            file = open(self.__models_save_folder_local.joinpath("global_model_parameters_%d.pkl" % (r+1)), "wb")
            pickle.dump(updated_global_model_parameters, file)
            file.close()

            log_print("[Server Info] Round %d finished! " % r, 'g')
            # ---------------------------------------------------------------------------- #

        # ----------------------------- Training finished ---------------------------- #
        create_file_on_local(path=str(self.__comm_channel_folder.joinpath('exit')))
        log_print("[Server Info] Training finished!", 'g')
        # ---------------------------------------------------------------------------- #
        return True