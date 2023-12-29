import pickle

runs = [27,28,29,30,31]

file_name_part_list = [
    "__20_player_per_game__100_indeptendnt_games__1_szenarios.pickle",
    "__20_player_per_game__100_indeptendnt_games__10_szenarios.pickle",
    "__20_player_per_game__100_indeptendnt_games__100_szenarios.pickle",
    "__500_player_per_game__1_indeptendnt_games__1_szenarios.pickle",
    "__500_player_per_game__1_indeptendnt_games__10_szenarios.pickle",
    "__500_player_per_game__1_indeptendnt_games__100_szenarios.pickle",
]


for file_name_part in file_name_part_list:

    pickle_dict_together = {
        "info_dict_list": [],
    }
    info_dict_list = []
    for run in runs:
        file = open("../saved_simulations/" + "run_"+str(run)+file_name_part, "rb")
        pickle_dict = pickle.load(file)
        info_dict_list += pickle_dict["info_dict_list"]
        pickle_dict_together = pickle_dict;

    pickle_dict_together["info_dict_list"]=info_dict_list
    file_to_save = open( "../saved_simulations/" + "all" + file_name_part, 'wb')
    pickle.dump(pickle_dict_together, file_to_save)



