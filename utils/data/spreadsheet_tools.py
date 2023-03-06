import pandas as pd


def combine_spreadsheets(spreadsheets, output_path):
    """Combine the rows of multiple Excel spreadsheets with the same columns,
    and save the resulting DataFrame to a specified file path.

    Args:
        spreadsheets (list): A list of file paths to the Excel spreadsheets.
        output_path (str): The file path to save the concatenated spreadsheet.

    Returns:
        None
    """
    # Load the first spreadsheet to initialize the DataFrame
    df = pd.read_excel(spreadsheets[0],  converters={'uid': str})

    # Concatenate the rows of all spreadsheets
    for path in spreadsheets[1:]:
        next_df = pd.read_excel(path, sheet_name=0,  converters={'uid': str})
        # print(next_df["uid"])
        df = pd.concat([df, next_df], ignore_index=False)

    # df["uid"] = df["uid"].apply('="{}"'.format)
    # Save the concatenated DataFrame to a new Excel file
    df.to_excel(output_path, index=False)


def combine_spreads():
    for fold in [0, 1, 2, 3]:
        spreadsheets = ['/mnt/tale_shared/schobs/landmark_unet/lannUnet_exps/GP/s1/f'+str(fold)+'_train/individual_results_fold'+str(fold)+'.xlsx',
                        '/mnt/tale_shared/schobs/landmark_unet/lannUnet_exps/GP/s1/f' +
                        str(fold)+'_val/individual_results_fold'+str(fold)+'.xlsx',
                        '/mnt/tale_shared/schobs/landmark_unet/lannUnet_exps/GP/s1/f'+str(fold)+'_test/individual_results_fold'+str(fold)+'.xlsx']

        output_path = '/mnt/tale_shared/schobs/landmark_unet/lannUnet_exps/GP/s1/s1_preds_all/combined_results_fold' + \
            str(fold)+'.xlsx'

        combine_spreadsheets(spreadsheets, output_path)
