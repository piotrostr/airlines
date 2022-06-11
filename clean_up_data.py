import pandas as pd

meaningful_cols = [
    'origin', 'destination', 'airline', 'booking_date', 'price', 'series',
    'countdown', 'rid', 'origin_name', 'type', 'lcc', 'hhi', 'mshare_rc',
    'legacy', 'category', 'extra', 'class', 'fare_class', 'flight_duration'
]


def clean_up_data():
    df = pd.read_stata('data.dta')
    df = df[meaningful_cols]
    df.write_csv('data.csv')


if __name__ == "__main__":
    print('cleaning up stata file')
    try:
        clean_up_data()
        print('done')
    except Exception as e:
        print('error:', e)
