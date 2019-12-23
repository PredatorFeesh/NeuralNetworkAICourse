#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <random>
#include <sstream>


namespace cifar
{
    using namespace std;

    class Cifar
    {

    public:

        typedef float dattype; 

        vector<dattype> y_train;
        vector<dattype> X_train;

        vector<dattype> y_test;
        vector<dattype> X_test;

        unsigned cifarNumImages = 1000;

        int size=32*32*3;

        Cifar(int num)
        {
            set_cifar(num);
        };
        ~Cifar(){};

        void set_cifar(int num) // X, y
        {
            // Read as such:
            // X -> all stacked on top of one another
            // Y -> also stacked on top of one another

            ifstream myfile ("cifar-10-binary/cifar-10-batches-bin/data_batch_"+to_string(num)+".bin",ios::binary);
            cout << endl;
            cout << "Making cifar" << endl;
            for (int w = 0; w < cifarNumImages; w++) 
            {   
                    
                    y_train.push_back( static_cast<uint8_t>(myfile.get()));     //binaryToDecimal()
                    for (unsigned i = 0; i < size; ++i) 
                    {
                        X_train.push_back( static_cast<dattype>(static_cast<uint8_t>(myfile.get())));  // binaryToDecimal()
                    }
            }

            cout << "Done making cifar" << endl;

            // return vector<vector<int>>{X_train, y_train};

        }

        void print(vector<dattype> vec)
        {
            for( dattype val : vec )
                cout << val << " ";
            cout << endl;
        } 


        vector<dattype> getonehot(int label, size_t max_labels)
        {
            vector<dattype> onehot(max_labels);

            for ( int i = 0 ; i < max_labels ; i++ )
                if (label == i)
                    onehot[i]=1;
                else
                    onehot[i]=0;
            
            return onehot;
        }

        vector<dattype> get_onehot(int image_number)
        {
            return getonehot(y_train[image_number], 10);
        } 

        vector<dattype> get_data(int image_number)
        {
            vector<dattype> data(size);
            int start_spot = image_number*size;
            for(int i = start_spot; i < (image_number+1)*size; i++)
                data[i - start_spot ] = X_train[i];
            return data;
        }



        vector<vector<dattype>> get_random(int type = 1)
        { // 0 = regular; 1 = onehot

            int randindx = rand() % (cifarNumImages);

            if ( type == 1 )
                return vector<vector<dattype>> { get_data(randindx), get_onehot(randindx) };
            else if ( type == 0 )
                return vector<vector<dattype>> { get_data(randindx), vector<dattype>{y_train[randindx]} };
        }

    };



}