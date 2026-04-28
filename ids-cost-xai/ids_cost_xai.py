import nslkdd
from logger import Logger

log = Logger()

def main():
    nslkdd_data = nslkdd.run_pipeline()
    # log.info("Train Data Sample:")
    # log.blank(f"\n{train_df.head()}")
    # log.info("Test Data Sample:")
    # log.blank(f"\n{test_df.head()}")

if __name__ == "__main__":
    main()