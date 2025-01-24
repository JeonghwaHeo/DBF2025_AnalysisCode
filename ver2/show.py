import argparse
from vsp_analysis import loadAnalysisResults, visualize_results
import pandas as pd

def get_result_by_id(resultID: int, csvPath: str="data/organized_results.csv") -> pd.DataFrame:
    resultID_df = pd.read_csv(csvPath, sep='|',encoding='utf-8')
    resultID_df = resultID_df[resultID_df['resultID'] == resultID]
    return resultID_df
    
    
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="module that displays the result screen which user wants.")
    subparsers = parser.add_subparsers(dest="main_command", required=True)
    show_parser = subparsers.add_parser("show", help="Show results for specific resultID.")
    
    show_subparsers = show_parser.add_subparsers(dest="type", required=True)
    show_aircraft_parser = show_subparsers.add_parser("aircraft", help="Show aircraft analysis results.")
    
    show_aircraft_parser.add_argument("resultID", type=int, help="Enter the resultID you want to check")
    
    args = parser.parse_args()
    if args.main_command == "show":
        if args.type == "aircraft":
            resultID_df = get_result_by_id(args.resultID)
            hashVal = resultID_df.loc[:,'hash']
            hashVal = hashVal.iloc[0]
            aircraft_result = loadAnalysisResults(hashVal)
            visualize_results(aircraft_result)
          
    

