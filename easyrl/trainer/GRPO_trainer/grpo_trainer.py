from easyrl.data_processor.training_data_processor import TrainingDataProcessor

def main():
    training_data_processor = TrainingDataProcessor(data_path='/run/determined/workdir/jiayanglyu/EasyRL/data/rl_train.parquet')
    while True:
        current_batch = training_data_processor.get_next_batch()
        print(len(current_batch[0]), len(current_batch))


if __name__ == "__main__":
    main()