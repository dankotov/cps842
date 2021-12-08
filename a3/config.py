import json

target_file = 'config.json'

with open(target_file, 'r') as config_file:
    config = json.load(config_file)

def change_stemming_option(new_value):
    config['enable_porter_stemmer'] = new_value
    with open(target_file, 'w') as config_file:
        json.dump(config, config_file)

def change_stop_word_removal_option(new_value):
    config['enable_stop_word_removal'] = new_value
    with open(target_file, 'w') as config_file:
        json.dump(config, config_file)
