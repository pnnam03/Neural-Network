#include <bits/stdc++.h>
using namespace std;
const int MAXN = 6e4 + 7;
unsigned int image[MAXN][30][30];
unsigned int num, magic, rows, cols;
unsigned int label[MAXN];
unsigned int in(ifstream &icin, unsigned int size)
{
    unsigned int ans = 0;
    for (int i = 0; i < size; i++)
    {
        unsigned char x;
        icin.read((char *)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}
void input()
{
    ifstream icin;
    icin.open("training-set/train-images.idx3-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);

    for (int i = 0; i < num; i++)
    {
        for (int x = 0; x < rows; x++)
        {
            for (int y = 0; y < cols; y++)
            {
                image[i][x][y] = in(icin, 1);
            }
        }
    }

    icin.close();
    icin.open("training-set/train-labels.idx1-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++)
    {
        label[i] = in(icin, 1);
    }
}

int main()
{
    input();
    ofstream outfile;
    outfile.open("training-set/training_input.dat");

    outfile << num << " " << rows << " " << cols << "\n";
    for (int i = 0; i < num; ++i)
    {
        outfile << label[i] << "\n";
        for (int x = 0; x < rows; ++x)
        {
            for (int y = 0; y < cols; ++y)
                outfile << image[i][x][y] << " ";
            outfile << "\n";
        }
    }
    outfile.close();
}