//
//  This Neural Network program is reading 1000 images from cifar10 dataset and training network for 10 images.
//  

#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <algorithm>

using namespace std;

const size_t cifarImageSize = (32 * 32 * 3);
const size_t cifarNumImages = 100;
const unsigned numLabels = 100;

void print ( const vector <float>& m, int n_rows, int n_columns ) 
{
    
    for( int i = 0; i != n_rows; ++i ) {
        for( int j = 0; j != n_columns; ++j ) {
            cout << m[ i * n_columns + j ] << " ";
        }
        cout << '\n';
    }
    cout << endl;
}

float getClosest(float, float, float); 

float findClosest(float arr[], int n, float target) 
{ 
  if (target <= arr[0]) 
    return arr[0]; 
  if (target >= arr[n - 1]) 
    return arr[n - 1]; 
  int i = 0, j = n;
        int mid = 0; 
  while (i < j) { 
    mid = (i + j) / 2; 

    if (arr[mid] == target) 
      return arr[mid]; 

    if (target < arr[mid]) 
                {
      if (mid > 0 && target > arr[mid - 1]) 
        return getClosest(arr[mid - 1],arr[mid], target); 
      j = mid; 
    } 

    else { 
      if (mid < n - 1 && target < arr[mid + 1]) 
        return getClosest(arr[mid], arr[mid + 1], target); 
      i = mid + 1; 
    } 
  } 
  return arr[mid]; 
} 

float getClosest(float val1, float val2, 
      float target) 
{ 
  if (target - val1 >= val2 - target) 
    return val2; 
  else
    return val1; 
} 


int argmax ( const vector <float>& m ) 
{

    return distance(m.begin(), max_element(m.begin(), m.end()));
}

vector <float> relu(const vector <float>& z){
    int size = z.size();
    vector <float> output;
    for( int i = 0; i < size; ++i ) {
        if (z[i] < 0){
            output.push_back(0.0);
        }
        else output.push_back(z[i]);
    }
    return output;
}

vector <float> reluPrime (const vector <float>& z) {
    int size = z.size();
    vector <float> output;
    for( int i = 0; i < size; ++i ) {
        if (z[i] <= 0){
            output.push_back(0.0);
        }
        else output.push_back(1.0);
    }
    return output;
} 

static vector<float> random_vector(const int size)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> distribution(0.0, 0.05);
    static default_random_engine generator;

    vector<float> data(size);
    generate(data.begin(), data.end(), [&]() { return distribution(generator); });
    return data;
}

float sigmoid(float x) 
{
   return (1.0 / (1.0 + std::exp(-x)));
}

float softmax (const vector <float>& z, const int dim) 
{
    
    const int zsize = static_cast<int>(z.size());
    float out;
    
    for (unsigned i = 0; i != zsize; i += dim) 
    {
        vector <float> foo;
        for (unsigned j = 0; j != dim; ++j) 
        {
            foo.push_back(z[i + j]);
        }
        
        float max_foo = *max_element(foo.begin(), foo.end());

        for (unsigned j = 0; j != dim; ++j) 
        {
            foo[j] = exp(foo[j] - max_foo);
        }      

        float sum_of_elems = 0.0;
        for (unsigned j = 0; j != dim; ++j) 
        {
            sum_of_elems = sum_of_elems + foo[j];
        }
        
        for (unsigned j = 0; j != dim; ++j) 
        {
            out = foo[j]/sum_of_elems;
        }
    }
    return out;
}

vector <float> sigmoid_d (const vector <float>& m1) 
{
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = m1[ i ] * (1 - m1[ i ]);
    }
    
    return output;
}

vector <float> sigmoid (const vector <float>& m1) 
{
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> output (VECTOR_SIZE);
    
    
    for( unsigned i = 0; i != VECTOR_SIZE; ++i ) {
        output[ i ] = 1 / (1 + exp(-m1[ i ]));
    }
    
    return output;
}

vector <float> operator+(const vector <float>& m1, const vector <float>& m2)
{
    
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> sum (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        sum[i] = m1[i] + m2[i];
    };
    
    return sum;
}

vector <float> operator-(const vector <float>& m1, const vector <float>& m2)
{
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> difference (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        difference[i] = m1[i] - m2[i];
    };
    
    return difference;
}

vector <float> operator*(const vector <float>& m1, const vector <float>& m2)
{
    const unsigned long VECTOR_SIZE = m1.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1[i] * m2[i];
    };
    
    return product;
}

vector <float> operator*(const float m1, const vector <float>& m2)
{
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m1 * m2[i];
    };
    
    return product;
}

vector <float> operator/(const vector <float>& m2, const float m1)
{    
    const unsigned long VECTOR_SIZE = m2.size();
    vector <float> product (VECTOR_SIZE);
    
    for (unsigned i = 0; i != VECTOR_SIZE; ++i){
        product[i] = m2[i] / m1;
    };
    
    return product;
}

vector <float> transpose (float *m, const int C, const int R) 
{
    vector <float> mT (C*R);
    
    for(unsigned n = 0; n != C*R; n++) {
        unsigned i = n/C;
        unsigned j = n%C;
        mT[n] = m[R*j + i];
    }
    
    return mT;
}

vector <float> dot (const vector <float>& m1, const vector <float>& m2, const int m1_rows, const int m1_columns, const int m2_columns) 
{    
    vector <float> output (m1_rows*m2_columns);
    
    for( int row = 0; row != m1_rows; ++row ) {
        for( int col = 0; col != m2_columns; ++col ) {
            output[ row * m2_columns + col ] = 0.f;
            for( int k = 0; k != m1_columns; ++k ) {
                output[ row * m2_columns + col ] += m1[ row * m1_columns + k ] * m2[ k * m2_columns + col ];
            }
        }
    }
    
    return output;
}


vector <float> subby(vector <float>& m1)
{
    for(int i=0;i<m1.size();i++)
     {
       m1[i]=1-m1[i]; 
     }
  
 
  return m1;
  
}

vector <float> minusme(vector <float>& m1)
{
    for(int i=0;i<m1.size();i++)
     {
       m1[i]=-m1[i]; 
     }
  
 
  return m1;
  
}

vector <float> findlog(vector <float>& m1, int k)
{
   if(k==0)
   {
     for(int i=0;i<m1.size();i++)
     {
       
          m1[i]=log(m1[i]);
     
     }
   }
   else
   {
      for(int i=0;i<m1.size();i++)
     {
      
          m1[i]=1-log(m1[i]);
      
     }
   }
  return m1;
  
}

vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

int binaryToDecimal(int n) 
{ 
    int num = n; 
    int dec_value = 0; 
  
    // Initializing base value to 1, i.e 2^0 
    int base = 1; 
  
    int temp = num; 
    while (temp) { 
        int last_digit = temp % 10; 
        temp = temp / 10; 
  
        dec_value += last_digit * base; 
  
        base = base * 2; 
    } 
  
    return dec_value; 
} 

int main(int argc, const char * argv[]) {

    string line;
    vector<string> line_v;

    cout << "Loading data ...\n";
    vector<float> X_train;
    vector<float> y_train;
    vector<float> b_X;
    vector<float> b_y;
    int randindx;
    vector<float> a1 ;
        vector<float> a2;
    vector<float> a3;
  vector<float>  a4;
        vector<float> yhat;
        vector<float> dyhat;
   
    ifstream myfile ("/home/honey/Documents/CS/fall2019-2nd sem/ai/proj/backpropagation-in-numpy-master/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin",ios::binary);
  
  int size=32*32*3;

   
 for (int w = 0; w < cifarNumImages; w++) 
  {   
         
           y_train.push_back( static_cast<uint8_t>(myfile.get()));     //binaryToDecimal()
           for (unsigned i = 0; i < size; ++i) 
          {
                X_train.push_back( static_cast<float>(static_cast<uint8_t>(myfile.get())));  // binaryToDecimal()
            
          }
    
  }
    
    int xsize = static_cast<int>(X_train.size());
    int ysize = static_cast<int>(y_train.size());
    
    // Some hyperparameters for the NN
    int BATCH_SIZE = 1;
    float lr = .01/BATCH_SIZE;

    // Random initialization of the weights
    vector <float> W1 = random_vector(3072*128);
    vector <float> W2 = random_vector(128*64);
    vector <float> W3 = random_vector(64*10);
     vector <float> B1;
    vector <float> B2 = random_vector(10) ;
    vector <float> dB1;
   vector <float> dB2;
    vector<float> dW1;
    vector<float> dW3;
    vector<float> dz2;
    vector<float> dW2 ;
    vector<float> dz1;
    vector<float> dy;
    float y;
    vector<float> yt;
    float ddy;
   float  n_c ;

    cout << "Training the model of size : "<<y_train.size()<<"\n";
    for (unsigned i = 0; i < 20; ++i)
    {

        randindx = rand() % (100-BATCH_SIZE);
   
           for (unsigned j = i*3072; j < (i+1)*3072; ++j)
        {
            b_X.push_back(X_train[i]);
        }

         unsigned k = randindx;
   
                 y = y_train[i];
          for(int m=0;m<10;m++)
          {
               if(m==y)
                 yt.push_back(1); 
              else
                 yt.push_back(0);
          }    

       // Feed forward
         a1 = relu(dot(  b_X, W1,BATCH_SIZE, 3072, 128 ));   
         a2 = relu(dot( a1, W2, BATCH_SIZE, 128, 64 )); 
         a3 = relu(dot( a2, W3, BATCH_SIZE, 64, 10 )); 
         a4 =  a3 + B2;  
        float arr[10];
        std::copy(a4.begin(), a4.end(), arr);
       // compute cost
    
    //     dy=sigmoid(findClosest(arr,10,y));
         dy=sigmoid(a4);
        // Back propagation
     
       //  dyhat.push_back(dy -  yt);
           dyhat = (dy -  yt);
        // dW3 = a2.T * dyhat
         dW3 = dot(transpose( &a2[0], BATCH_SIZE, 64 ), dyhat, 64, BATCH_SIZE, 10);  
        // dz2 = dyhat * W3.T * relu'(a2)
        dz2 = dot(dyhat, transpose( &W3[0], 64, 10 ), BATCH_SIZE, 10,64) *  reluPrime(a2); 
        // dW2 = a1.T * dz2
         dW2 = dot(transpose( &a1[0], BATCH_SIZE, 128 ), dz2, 128, BATCH_SIZE, 64);  
        // dz1 = dz2 * W2.T * relu'(a1)
         dz1 = dot(dz2, transpose( &W2[0], 128, 64 ), BATCH_SIZE, 64,128) *  reluPrime(a1); 
        // dW1 = X.T * dz1
         dW1 = dot(transpose( &b_X[0], BATCH_SIZE, 32*32*3), dz1, 32*32*3, BATCH_SIZE, 128);  
         dB2=dz2;
         dB1=dz1; 
        // Updating the parameters
        W3 = W3 - lr * dW3;
        W2 = W2 - lr * dW2;
        W1 = W1 - lr * dW1;
        B2 = B2 - lr * dB2; 
        B1 = B1 - lr * dB1; 

       
       //  cout << "Predictions: " <<dy<< "\n";
       // cout << "Original : " <<y<< "\n";

       //   if (dy > 0.5)
        //          n_c += 1;

        if ((i+1) % 10 == 0)
        {
            cout << "-------------------Epoch " << i+1 << "--------------------"<<"\n";
            cout << " Final Predictions:" << "\n";
            print ( dy, 10, 1 );
        //   cout<<dy;
            cout << "\nGround truth:" << "\n";
            //cout<<y;
            print(yt,10,1);            
            vector <float> loss_m = dy - yt;
            float loss = 0.0;
            for(int m=0;m<10;m++)          
                loss += loss_m[m]*loss_m[m]; 
            cout << " \nAccuracy " << loss/i<<"\n";     //BATCH_SIZE
            cout << "------------------End of Epoch :(-----------------------" <<"\n";
        };
        b_X.clear();
        yt.clear();
    };
    
    // Testing 1 image at loc 20

        for (unsigned j = 40*3072; j < (40+BATCH_SIZE)*3072; ++j)
        {
            b_X.push_back(X_train[j]);
        }
  
            y = y_train[40];
       
    
        a1 = (dot(  b_X, W1,BATCH_SIZE, 3072, 128 ));   
        a2 = (dot( a1, W2, BATCH_SIZE, 128, 64 )); 
       
        a3 = relu(dot( a2, W3, BATCH_SIZE, 64, 10 )); 
         a4 =  a3 + B2;  
          float arr[10];
        std::copy(a4.begin(), a4.end(), arr);
       
        // dy=sigmoid(findClosest(arr,10,y)); 
           dy=sigmoid(a4); 
        cout << "\nTest Predictions:" << "\n";
        print(dy,10,1);
        cout << "\nGround truth:" << "\n";
        print(yt,10,1); 
       vector <float> loss_m = dy - yt;
            float loss = 0.0;
            for(int m=0;m<10;m++)          
                loss += loss_m[m]*loss_m[m]; 
            cout << " \nAccuracy " << loss/20<<"\n"; 

    return 0;
}

