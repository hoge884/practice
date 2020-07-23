/*
    this program will do 1-D POC function and estimate amount of movement between 2 pictures
    use 2 fictures and read as grayscale so estimate in grayscale

    I've already finished making this program when I was university student
    so I'd like to change this program to C++ and check how C++ is better than C to make sure
*/

// インクルード-----------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cstdlib>
#include <vector>
#include <complex>
#include "applywindow.hpp"
#include "dft.hpp"
#include "cps.hpp"
#include "idft.hpp"
// インクルード終了------------------------------------------------------------------------------------------

#define Y 8
#define X 3

// 名前空間の定義--------------------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
// 名前空間の定義終了----------------------------------------------------------------------------------------------


// メイン
int main(int argc, const char* argv[]) {
    /* read image here */
    Mat mat = imread("sample7_17.png", 0);
    Mat mat2 = imread("sample7_17.png", 0);

    /* get height and width */
    int height = mat.rows;
    int width = mat.cols;
    
    /* if not read image, set pixel value manually */
    Mat result = Mat_<complex<double> >(height,width);
    Mat result2 = Mat_<complex<double> >(height,width);
    Mat idftResult = Mat_<complex<double> >(height,width);
    Mat idftResult2 = Mat_<complex<double> >(height,width);
    Mat finalResult = Mat_<complex<double> >(1,width);


    double checker = 0.0;

    // POC結果の平均を求める際に使う
    complex<double> ave = (0,0);

    /* check height and width to make sure */
    cout << "高さは　" << height << endl;
    cout << "幅は　" << width << endl;


    /* convert from char to float 'cause window functions has calculation formula in float */
    mat.convertTo(mat,CV_64F,1.0/255);
    mat2.convertTo(mat2, CV_64F, 1.0/255);

    // 今回はテストとして、画像を白色画像に手動で変えてある
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            mat.at<double>(y,x) = 1.0;
            mat2.at<double>(y,x) = 1.0;
        }
    }

    
    /* apply window function here */
    ApplyWin *win;
    win = new ApplyWin();
    win -> Hamming(mat, Y,X);
    win -> Hamming(mat2, Y,X);
    delete win;

    /* applyDFT here */
    DFT *dft;
    dft = new DFT();
    dft -> applyDFT(mat, result, Y, X);
    dft -> applyDFT(mat2, result2, Y, X);
    delete dft;

    /* calc CPS here */
    CPS *cps;
    cps = new CPS();
    cps -> addCPS(result, result2, Y, X);
    delete  cps;

    /* apply IDFT here */
    IDFT *idft;
    idft = new IDFT();
    idft -> applyIDFT(idftResult, result, Y, X);
    idft -> applyIDFT(idftResult2, result2, Y, X);
    delete  idft;

    /*
        ここでPOC結果の平均を求めている
    */
    for (int x = 0; x < mat.cols; x++) {
        for (int y = 0; y < mat.rows; y++) {
            ave += idftResult.at<complex<double> >(y,x);
        }
        ave /= height;
        finalResult.at<complex<double> >(0,x) = ave;
        ave = (0,0);
    }



    /*
        csv形式のファイルを作成する処理
    */
    ofstream ofs_csv_file("test.csv");

    ofs_csv_file << -3 << ',' << finalResult.at<complex<double> >(0.0) << endl;
    ofs_csv_file << -2 << ',' << finalResult.at<complex<double> >(0,1) << endl;
    ofs_csv_file << -1 << ',' << finalResult.at<complex<double> >(0,2) << endl;
    ofs_csv_file << 0 << ',' << finalResult.at<complex<double> >(0,3) << endl;
    ofs_csv_file << 1 << ',' << finalResult.at<complex<double> >(0,4) << endl;
    ofs_csv_file << 2 << ',' << finalResult.at<complex<double> >(0,5) << endl;
    ofs_csv_file << 3 << ',' << finalResult.at<complex<double> >(0,6) << endl;




    /* show all elements of mat */
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            cout << "(" << y << " , " << x << ")" << " = " << idftResult.at<complex<double> >(y,x) << endl;
        }
    }

    for (int x = 0; x < width; x++) {
        cout << "(1"  << " , " << x << ")" << " = " << finalResult.at<complex<double> >(0,x) << endl;
    }





    /* make window named what between " is */
    //namedWindow("test");
    /* keep picture function */
    //imwrite("test.png", mat);
    /* show image function */
    //imshow("test",mat);
    /* wait user inputting anykeys */
    //waitKey(0);

    
    return 0;
}