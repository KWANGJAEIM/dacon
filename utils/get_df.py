import pandas as pd 

#file_path가 기본적으로 리눅스에 맞게 되어있어서 윈도우스로 맞춰주기 위해 
CFG = {
    'VIDEO_LENGTH':50, # 10프레임 * 5초
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':32,
    'SEED':41,
    "ROOT_DIR": "c:\\Users\\windowadmin5\\Desktop\\kj\\ActionClassification\\Dataset",
    "check_dir": "c:\\Users\\windowadmin5\\Desktop\\kj\\ActionClassification\\checkpoint\\",
    "model_name":"3d_resnet"
}
def get_df(file_path):
    df = pd.read_csv(file_path)
    df["video_path"] = list(map(lambda x: x.split(".")[1]+"."+x.split(".")[-1],df["video_path"]))
    df["video_path"]= list(map(lambda x:CFG["ROOT_DIR"]+x, df["video_path"]))

    return df 

def get_label(df):
    new_label=pd.DataFrame(columns =["crash", "ego_involve", "weather", "timing"])
    for i in len(df):
        if df["label"].iloc[i]==0:
        
        else: 
            new_label["crash"]=1
            if df["label"].iloc[i] in [1,2,3,4,5,6]:
                new_label["ego-involve"]=1
            else:
                 new_label["ego-involve"]=1