from torch.utils.data import Dataset
import random

class AmazonDataset(Dataset):

    def __init__(self, file_path, transform=None):
        self.transform = transform
        self.size = 0
        with open(file_path, 'r') as input:

            meta = input.readline().split(',')
            self.size = int(meta[0])
            self.feature_dim = int(meta[1])
            
            # self.size = 1024 # this is just for debug the dataLoader
            print("Start to processing <"+ file_path+"> with "+str(self.size)+ " samples.")
            self.label_set = []
            self.query_feature_index_set = []
            self.query_feature_value_set = []
            self.title_feature_index_set = []
            self.title_feature_value_set = []
            for i in range(self.size):
                if i % (self.size//10)==0:
                    print((100*i//self.size + 1), "% is done.")
                #print (input.readline())
                items = input.readline().split(',')
                if len(items) < 3:
                    continue
                #while len(items) < 3:
                #   items = input.readline().split(',')
                #   print(items)
                
                label = items[0]
                self.label_set.append(int(label))
                query_feature_value = []
                query_feature_index = []
                title_feature_value = []
                title_feature_index = []

                
                query_items = items[1].split(' ')
                #print (query_items)
                # We iterate from 1 to N-1 to avoid two empty items
                for ii in range(1, len(query_items)-1):
                    #print (query_items[ii])
                    index, value = query_items[ii].split(':')
                    query_feature_index.append(int(index))
                    query_feature_value.append(float(value))

                self.query_feature_index_set.append(query_feature_index)
                self.query_feature_value_set.append(query_feature_value)
                
                title_items = items[2].split(' ')
                #print (title_items)
                for ii in range(0, len(title_items)-1):
                    index, value = title_items[ii].split(':')
                    title_feature_index.append(int(index))
                    title_feature_value.append(float(value))

                self.title_feature_index_set.append(title_feature_index)
                self.title_feature_value_set.append(title_feature_value)
            #Generate Negative Samples (Randomly Generating Samples) 
            print ("Generating Negative Samples!")
            for i in range(self.size):
                if i % (self.size//10)==0:
                    print((100*i//self.size + 1), "% is done.")
                
                self.query_feature_index_set.append(self.query_feature_index_set[i])
                self.query_feature_value_set.append(self.query_feature_value_set[i])

                neg_idx = random.randint(0, self.size-1)
                self.title_feature_index_set.append(self.title_feature_index_set[neg_idx])
                self.title_feature_value_set.append(self.title_feature_value_set[neg_idx])
               
                self.label_set.append(-1)
            
            self.size += self.size

 

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {'label':self.label_set[index],
                  'query_feature_value': self.query_feature_value_set[index],
                  'query_feature_index': self.query_feature_index_set[index],
                  'title_feature_index': self.title_feature_index_set[index],
                  'title_feature_value': self.title_feature_value_set[index],
                  }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_feature_dim(self):
        return self.feature_dim

#We might not need this I guess, so let's leave it here

def AmazonDataset_collate(batch):
    batch_size = len(batch)
    query_feature_index_batch = [[],[]]
    query_feature_value_batch = []

    title_feature_index_batch = [[],[]]
    title_feature_value_batch = []
   
    label_batch = []
    for i in range(batch_size):
        
        #query_feature_index_batch.append(batch[i]['query_feature_index'])
        #title_feature_index_batch.append(batch[i]['title_feature_index'])

        query_feature_value_batch.extend(batch[i]['query_feature_value'])
        query_feature_index_batch[0].extend([i]* len(batch[i]['query_feature_index']))
        query_feature_index_batch[1].extend(batch[i]['query_feature_index'])
        title_feature_value_batch.extend(batch[i]['title_feature_value'])
        title_feature_index_batch[0].extend([i]* len(batch[i]['title_feature_index']))
        title_feature_index_batch[1].extend(batch[i]['title_feature_index']) 
        
        
        label_batch.append(batch[i]['label'])
        '''
    result = {'query_feature_index_batch':query_feature_index_batch,
            'title_feature_index_batch':title_feature_index_batch, 
            'label':label_batch}
        '''  
    result = {'query_feature_index_batch':query_feature_index_batch,        
            'query_feature_value_batch':query_feature_value_batch,
            'title_feature_index_batch':title_feature_index_batch, 
            'title_feature_value_batch':title_feature_value_batch,
            'label':label_batch}
    #print(result)
    return result


def train_dataset(data_path = '/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_test_sparse.data'):
    return AmazonDataset(data_path)


def test_dataset(data_path='/home/cl67/workspace/se/Amazon-Product-Search-Datasets/reviews_CDs_and_Vinyl_5.json.gz/query_asin_train_sparse.data'):
    return AmazonDataset(data_path)


def main():
    #train_set = train_dataset()
    test_set = test_dataset()


if __name__ == '__main__':
    main()
