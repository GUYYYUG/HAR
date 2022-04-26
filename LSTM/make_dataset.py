

import pickle

import numpy as np
# raw_data_name = "./UCI HAR Dataset/train/X_train.txt"

def pre_process_data(raw_data_name,label_name):
    ans = []
    with open(raw_data_name,"r") as f:
        
        for line in f.readlines():
            content = line.strip().split(' ')
            for item in content[:]:
                if item == " " or item == '':
                    content.remove(item)
            
            
            # print(len(content))
            seq = [float(e) for e in content]
            ans.append({'data':seq})
            
        # ans = np.asarray(ans)
        # print(ans.shape)
    with open(label_name,"r") as f:
        iterx = 0
        for line in f.readlines():
            content = line.strip()
            res = float(content)
            ans[iterx]['label'] = res
            iterx = iterx + 1

    print(len(ans))
    return ans

if __name__ == '__main__':
    raw_data_name = "./UCI HAR Dataset/test/X_test.txt"
    label_name = "./UCI HAR Dataset/test/y_test.txt"
    train = pre_process_data(raw_data_name,label_name)
    # with open('test.pkl','wb') as f:
    #     pickle.dump(train,f)
    
    with open('test.pkl','rb') as f:  # Python 3: open(..., 'rb')
        item = pickle.load(f)
        print(item[0]['data'])


