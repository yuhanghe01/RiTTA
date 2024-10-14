import yaml
import os
import TextAudioCreator

if __name__ == '__main__':
    yaml_config_filename = 'tta_datagen_config.yaml'
    assert os.path.exists(yaml_config_filename)
    with open(yaml_config_filename) as f:
        config = yaml.safe_load(f)
    text_audio_creator = TextAudioCreator.TextAudioCreator(config)
    text_audio_creator.get_textprompt_and_audio()