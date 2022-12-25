"""
Wikidata utility
"""
import random
import copy
import urllib.request
import json
from multiprocessing import Process, Manager, Pool
import multiprocessing
import math
import socket
import time
from tqdm import tqdm
import os
from datetime import datetime, date
from pymongo import MongoClient


class WikidataDumpUtils:
    """
    utility:
    (1) load part of wikidata json file into database (related to the relation we care about)
    """
    def __init__(self, split_data_folder, entity_id_list_file):
        self.split_data_folder = split_data_folder
        self.entity_id_list_file = entity_id_list_file

    @staticmethod
    def _check_if_this_obj_is_human(json_obj):
        type_set = set()
        try:
            for relation_id, info_list in json_obj["claims"].items():
                if relation_id == "P31":
                    for info_item in info_list:
                        type_value = info_item["mainsnak"]["datavalue"]["value"][
                            "id"]
                        type_set.add(type_value)
                    # endfor
                # endif
            # endfor
        except:
            pass
        finally:
            if "Q5" in type_set:
                return True
            else:
                return False
            # endif
        #endtry

    def collect_file_path_pool(self, folder):
        file_path_list = []
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            file_path_list.append(file_path)
        # endfor
        return file_path_list

    def _search_wikidata_id_list_in_one_file(self, file_path_list,
                                             wikidata_id_set):

        # Creating a pymongo client
        client = MongoClient('localhost', 27017)

        # Getting the database instance
        db = client['wikidata_entities_subset_431_relation_all']
        print("Database created........")

        # Verification
        print("List of databases after creating new one")
        print(client.list_database_names())

        for file_path in file_path_list:
            with open(file_path, mode="r") as fin:
                for line in fin:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    # endif
                    json_obj = json.loads(line)
                    entity_id = json_obj["id"]
                    if entity_id in wikidata_id_set:
                        result_list = []
                        for tmp_obj in db.entity_obj.find({"id": entity_id}):
                            result_list.append(tmp_obj)
                        # endfor

                        if len(result_list) == 0:
                            db.entity_obj.insert_one(json_obj)
                            print(f"{entity_id} inserted.")
                    # endif
                # endfor
            # endwith
        # endfor

    def parallelly_insert_entity_id_into_mongodb(self):
        """
        Insert all entity json into database
        """
        # ######## load entity id set #########
        all_entity_id_set = set()
        with open(self.entity_id_list_file, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                all_entity_id_set.add(line)
            # endfor
        # endwith
        print(
            f"There are {len(all_entity_id_set)} entity id need to insert into database."
        )

        num_of_worker = 60
        file_path_list = self.collect_file_path_pool(self.split_data_folder)
        total_file_path_num = len(file_path_list)
        print(
            f"There are totally {total_file_path_num} files to be extracted, "
            f"Each file contains 5000 json objects, except the last one.")
        block_size = math.ceil(total_file_path_num / num_of_worker)

        # allocate tasks
        pool = Pool(processes=num_of_worker)
        workers = []
        for w in range(num_of_worker):
            print(f"worker {w} starts ...")
            block_list = file_path_list[w * block_size:(w + 1) * block_size]
            single_worker = pool.apply_async(
                func=self._search_wikidata_id_list_in_one_file,
                args=(block_list, all_entity_id_set))

            workers.append(single_worker)
        # endfor
        pool.close()

        for w in tqdm(workers, desc="all jobs"):
            w.get()
        # endfor

        print("Finished.")

    @staticmethod
    def _single_worker_check_whether_entity_id_set_in_database(
            entity_id_list, shared_entity_id_not_in_database_list):

        # Creating a pymongo client
        client = MongoClient('localhost', 27017)

        # Getting the database instance
        db = client['wikidata_dump']
        # print("Database created........")

        for entity_id in entity_id_list:

            result_list = []
            for tmp_obj in db.entity_obj.find({"id": entity_id}):
                result_list.append(tmp_obj)
            # endfor

            if len(result_list) == 0:
                shared_entity_id_not_in_database_list.append(entity_id)
                print(f"{entity_id} not in database.")
                continue
            else:

                # (1) check if it is human
                entity_obj = result_list[0]
                is_human = WikidataDumpUtils._check_if_this_obj_is_human(
                    entity_obj)
                if is_human:
                    continue
                else:
                    shared_entity_id_not_in_database_list.append(entity_id)
                    print(f"{entity_id} does not contain P31 Q5")
                #endif

                # (2) check if it contains english label (the human entity should contain english label)
                try:
                    label = entity_obj["labels"]["en"]["value"]
                    if len(label) > 0:
                        continue
                    else:
                        shared_entity_id_not_in_database_list.append(entity_id)
                    #endif
                except:
                    print(f"{entity_id} does not contain English label.")
                    shared_entity_id_not_in_database_list.append(entity_id)
                    continue
                #endtry
            #endif


        # endfor
        pass

    def parallel_get_human_id_list_not_in_high_quality(self):
        # ######## load entity id set #########
        all_entity_id_set = set()
        with open(self.entity_id_list_file, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                # endif
                all_entity_id_set.add(line)
            # endfor
        # endwith
        print(
            f"There are {len(all_entity_id_set)} entity id need to insert into database."
        )
        all_entity_id_list = list(all_entity_id_set)

        num_of_worker = 60
        total_entity_id_num = len(all_entity_id_list)
        print(
            f"There are totally {total_entity_id_num} entity id need to be validated."
        )
        block_size = math.ceil(total_entity_id_num / num_of_worker)

        # shared entity id set not in database
        manager = Manager()
        shared_entity_id_set_not_in_database = manager.list()

        print(f"building pool and setting up {num_of_worker} workers ...")
        pool = Pool(processes=num_of_worker)

        workers = []
        for w in range(num_of_worker):
            print(f"worker {w} starts ...")
            block_list = all_entity_id_list[w * block_size:(w + 1) *
                                            block_size]
            single_worker = pool.apply_async(
                func=WikidataDumpUtils._single_worker_check_whether_entity_id_set_in_database,
                args=(block_list, shared_entity_id_set_not_in_database))

            workers.append(single_worker)
        # endfor
        pool.close()

        for w in tqdm(workers, desc="all jobs"):
            w.get()
        # endfor

        entity_id_not_in_database_set = set(shared_entity_id_set_not_in_database)

        from datetime import datetime
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

        with open(f"entity_id_not_in_database_set_{date_time}.txt",
                  mode="w") as fout:
            for item in entity_id_not_in_database_set:
                fout.write(f"{item}\n")
            #endfor
        #endwith

        print("Finished.")
        pass

    def _single_insert_all_json_obj_in_file_into_database(
            self, file_path_list, worker_id):
        # Creating a pymongo client
        client = MongoClient('localhost', 27017)

        # Getting the database instance
        db = client['wikidata_dump']
        print("Database created........")

        # Verification
        print("List of databases after creating new one")
        print(client.list_database_names())

        insert_block = []
        index = 0
        for file_path in tqdm(file_path_list, desc=f"worker-{worker_id}"):
            with open(file_path, mode="r") as fin:
                for line in fin:
                    index += 1
                    if index % 100 == 0:
                        print(".", end="")
                    #endif
                    if index % 10000 == 0:
                        print("\n")
                    #endif
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    # endif

                    try:
                        json_obj = json.loads(line)
                    except:
                        print(f"ERROR: {line}")
                        continue
                    #endtry

                    entity_id = json_obj["id"]
                    # check if it is in the database
                    result_list = []
                    for tmp_obj in db.entity_obj.find({"id": entity_id}):
                        result_list.append(tmp_obj)
                    # endfor

                    if len(result_list) == 0:
                        insert_block.append(json_obj)
                    #endif

                    if len(insert_block) == 1000:
                        db.entity_obj.insert_many(insert_block)
                        print(
                            f"worker-{worker_id}: {len(insert_block)} inserted."
                        )
                        insert_block = []
                    #endif

                # endfor
            # endwith
        # endfor
        if len(insert_block) > 0:
            db.entity_obj.insert_many(insert_block)
            print(f"worker-{worker_id}: {len(insert_block)} inserted.")
        #endif
        pass

    def parallelly_insert_all_wikidata_json_into_database(self):
        """
        Insert all entity json into database

        There are totally 18597 files to be extracted, Each file contains 5000 json objects, except the last one.
        """
        num_of_worker = 60
        file_path_list = self.collect_file_path_pool(self.split_data_folder)
        random.shuffle(file_path_list)

        total_file_path_num = len(file_path_list)
        print(
            f"There are totally {total_file_path_num} files to be extracted, "
            f"Each file contains 5000 json objects, except the last one.")
        block_size = math.ceil(total_file_path_num / num_of_worker)

        # allocate tasks
        pool = Pool(processes=num_of_worker)
        workers = []
        for w in range(num_of_worker):
            print(f"worker {w} starts ...")
            block_list = file_path_list[w * block_size:(w + 1) * block_size]
            single_worker = pool.apply_async(
                func=self._single_insert_all_json_obj_in_file_into_database,
                args=(block_list, w))

            workers.append(single_worker)
        # endfor
        pool.close()

        for w in tqdm(workers, desc="all jobs"):
            w.get()
        # endfor
        print("Finished.")
        pass

    def parallelly_insert_all_wikidata_json_into_database_from_file_path_list(
            self, file_path_list):
        """
        Insert all entity json into database

        There are totally 18597 files to be extracted, Each file contains 5000 json objects, except the last one.
        """
        num_of_worker = 60
        random.shuffle(file_path_list)

        total_file_path_num = len(file_path_list)
        print(
            f"There are totally {total_file_path_num} files to be extracted, "
            f"Each file contains 5000 json objects, except the last one.")
        block_size = math.ceil(total_file_path_num / num_of_worker)

        # allocate tasks
        pool = Pool(processes=num_of_worker)
        workers = []
        for w in range(num_of_worker):
            print(f"worker {w} starts ...")
            block_list = file_path_list[w * block_size:(w + 1) * block_size]
            single_worker = pool.apply_async(
                func=self._single_insert_all_json_obj_in_file_into_database,
                args=(block_list, w))

            workers.append(single_worker)
        # endfor
        pool.close()

        for w in tqdm(workers, desc="all jobs"):
            w.get()
        # endfor
        print("Finished.")
        pass


class WikidataUtils:
    @staticmethod
    def download_single_entry_online(entry_id):
        """
        The entry id can be
        1. entity id: such as Q42
        2. property id: such as P991

        Both entities and properties can be downloaded using the same URL pattern
        https://www.wikidata.org/wiki/Special:EntityData/Q42.json
        https://www.wikidata.org/wiki/Special:EntityData/P991.json

        Note: For some reason, when VPN is connected, it is super slow, nearly impossible to download json obj

        Crawler Restriction:
        retrieve data from wikipedia, the same for wikidata: https://en.wikipedia.org/wiki/Wikipedia:Database_download
        Please do not use a web crawler -> Sample blocked crawler email
        Something like at least a second delay between requests is reasonable.
        add
        ```
        time.sleep(1.5)
        ```

        Note that: wikidata is updating and changing, some entity id may be redirected to other id.
        For instance, try to download: Q104295373 using URL: https://www.wikidata.org/wiki/Q104295373
        it will redirect to -> https://www.wikidata.org/wiki/Q29939705

        In this case:
        (1) deep copy the download json object and change the download json object id to query id
        (2) change return two json object, and insert to database for both of the json object.


        # check: 'Q104295373'
        # 'Q29939705'

        """
        # date
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")

        time.sleep(1.5)
        URL = f"https://www.wikidata.org/wiki/Special:EntityData/{entry_id}.json"
        print(f"downloading from url: {URL}")

        try_count = 0
        data_str = None
        json_obj = None
        while data_str is None and try_count <= 5:
            try:
                data_str = urllib.request.urlopen(URL).read().decode()
                # parse json object
                json_obj = json.loads(data_str)

                for k, v in json_obj["entities"].items():
                    json_obj_ready = v
                # endfor

                # json object id
                cur_entity_id = json_obj_ready["id"]

                if cur_entity_id != entry_id:
                    old_json_obj = copy.deepcopy(json_obj_ready)
                    old_json_obj["id"] = entry_id
                
                    return [json_obj_ready, old_json_obj]
                else:
                    return [json_obj_ready]


            except Exception as e:
                print(e)
                with open(
                        f"WikidataUtils_download_single_entry_online_ERROR_{d1}.log",
                        mode="a") as fout:
                    now = datetime.now()
                    dt_string = now.strftime("%d-%m-%Y-%H:%M:%S")
                    fout.write(
                        f"{dt_string} ERROR: {URL} cannot be retrieved from wikidata. \n"
                    )
                    try_count += 1
                # endwith
                print("trying again ...")
            # endtry
        # endwhile

        return None


def main_download_and_insert_entity_into_mongodb():
    # Creating a pymongo client
    client = MongoClient('localhost', 27017)

    # Getting the database instance
    db = client['wikidata_entities_subset_431_relation_all']
    print("Database created........")

    # Verification
    print("List of databases after creating new one")
    print(client.list_database_names())

    # ###################### input ########################
    input_entity_id_list = []
    with open("tmp_related_wikidata_entity_id_list.txt", mode="r") as fin:
        for line in fin:
            line = line.strip()
            input_entity_id_list.append(line)
        # endfor
    # endwith

    for entity_id in input_entity_id_list:
        result_list = []
        for item in db.entity_obj.find({"id": entity_id}):
            result_list.append(item)
        # endfor
        if len(result_list) == 0:
            download_json = WikidataUtils.download_single_entry_online(
                entity_id)

            if download_json is not None:
                for k, v in download_json["entities"].items():
                    json_obj_ready = v
                # endfor

                db.entity_obj.insert_one(json_obj_ready)
                print(f"property {entity_id} is inserted.")
            # endif
    # endfor


def main_search_wikidata_dump_load_entity_obj_into_mongodb():
    """
    Given entity id list, traverse wikidata dump and load into mongodb
    """
    # wikidata id list file
    wikidata_id_list_file = "../data/merged_human_id_set.txt"

    # wikidata_utils
    wikidata_processor = WikidataDumpUtils(
        split_data_folder=
        "/hdd_1/nma4/python_project/Name_Entity_Novelty_Detection_Dataset_Preparation/data/wikidata/wikidata_json_split_fast",
        entity_id_list_file=wikidata_id_list_file)

    wikidata_processor.parallelly_insert_entity_id_into_mongodb()
    pass


def main_insert_all_wikidata_object_into_mongodb():
    """
    Insert all wikidata dump into mongodb
    """
    # wikidata_utils
    wikidata_processor = WikidataDumpUtils(
        split_data_folder=
        "/hdd_1/nma4/python_project/Name_Entity_Novelty_Detection_Dataset_Preparation/data/wikidata/wikidata_json_split_fast",
        entity_id_list_file=None)

    wikidata_processor.parallelly_insert_all_wikidata_json_into_database()
    pass


def main_insert_extra_human_obj_from_sparql_query():
    """
    Insert extra sparql human object into database
    """
    wikidata_processor = WikidataDumpUtils(split_data_folder=None,
                                           entity_id_list_file=None)

    file_path = "/hdd_1/nma4/python_project/Name_Entity_Novelty_Detection_Dataset_Preparation/data/wikidata/3_sparql_human_json_file/merged_sparql_human_json_file.txt"
    file_path_list = [file_path]
    wikidata_processor.parallelly_insert_all_wikidata_json_into_database_from_file_path_list(
        file_path_list)


def filter_out_low_quality_human_id_from_database():
    """
    Human id contains two parts:
    (1) human id in wikidata dump
    (2) human id from sparql that are not in wikidata, which need to download json and store to database

    # this function make sure that:
    (1) human id in database
    (2) the wikidata obj contains (-, P31, Q5)
    
    Output
    =======
    entity_id_not_in_database_set_12-27-2021-17-34-17.txt  ->  set of size 14860
    entity_id_not_in_database_set_12-28-2021-14-41-21.txt  -> 14860
    
    I have double checked that, the human id in this file cannot be found in wikidata online
    So, currently, database contains all the human ids.

    """
    # wikidata id list file
    wikidata_id_list_file = "../data/merged_human_id_set.txt"

    # wikidata_utils
    wikidata_processor = WikidataDumpUtils(
        split_data_folder=None, entity_id_list_file=wikidata_id_list_file)

    wikidata_processor.parallel_get_human_id_list_not_in_high_quality()

    pass


def get_high_quality_human_id_list():
    """
    (1) original human id set:
    "../data/merged_human_id_set.txt"

    (2) low quality human id set file
    entity_id_not_in_database_set_12-27-2021-17-34-17.txt

    output
    ========
    high_quality_human_id_set.txt

    -----
    original human id set: 9238907
    low quality human id set: 14860
    high quality human id set: 9224047

    """
    def load_human_id_set(file_path):
        human_id_set = set()
        with open(file_path, mode="r") as fin:
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    continue
                #endif
                human_id_set.add(line)
            #endfor
        #endwith
        return human_id_set

    
    # (1)
    original_human_id_set_file = "../data/merged_human_id_set.txt"
    # (2)
    low_quality_human_id_set_file = "entity_id_not_in_database_set_12-27-2021-17-34-17.txt"

    original_human_id_set = load_human_id_set(original_human_id_set_file)
    low_quality_human_id_set = load_human_id_set(low_quality_human_id_set_file)
    high_quality_human_id_set = original_human_id_set - low_quality_human_id_set
    
    print(f"original human id set: {len(original_human_id_set)}")
    print(f"low quality human id set: {len(low_quality_human_id_set)}")
    print(f"high quality human id set: {len(high_quality_human_id_set)}")

    with open("high_quality_human_id_set.txt", mode="w") as fout:
        for human_id in high_quality_human_id_set:
            fout.write(f"{human_id}\n")
        #endfor
    #endwith



if __name__ == '__main__':
    # entity_id = "Q42"
    # print(WikidataUtils.download_single_entry_online(entity_id))

    # (1)
    # main_download_and_insert_entity_into_mongodb()

    # (2)
    # main_search_wikidata_dump_load_entity_obj_into_mongodb()

    # (3)
    # main_insert_all_wikidata_object_into_mongodb()

    # (4)
    # main_insert_extra_human_obj_from_sparql_query()

    # (5)
    filter_out_low_quality_human_id_from_database()

    # (6)
    # get_high_quality_human_id_list()
    pass
