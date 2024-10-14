import yaml
import EmbedExtractor

if __name__ == '__main__':
    config_filename = 'eval_config.yaml'
    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    extractor = EmbedExtractor.EmbedExtractor(config)
    audio_data_dir = '/mnt/nas/yuhang/audioldm/tango-finetuned'
    embed_type = ['vggish', 'panns']
    extractor.get_embedding(audio_data_dir, embed_type)