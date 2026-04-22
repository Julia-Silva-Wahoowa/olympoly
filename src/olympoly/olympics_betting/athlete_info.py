# Hans and Lauren
# script to standardize string input with pandas function
# return athlete data

from olympoly.load_data import load_olympic_data

def get_athlete_info(athlete_name: str):
    df = load_olympic_data()
    df["Name"] = df["Name"].str.split(",").str[0]
    athlete_info = df[df["Name"].str.contains(athlete_name, case=False)]
    print(athlete_info)
    return None

#print(get_athlete_info("Michael Fred Phelps"))