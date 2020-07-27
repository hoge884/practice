/*

    このプログラムは「あ」から「お」までのあ行の文字認識を行う
    濃度、水平方向ラン数、垂直方向ラン数、外接矩形比の4つの特徴量を抽出して比較する

    画像の読み込みなどにはOpenCVの機能を用いる。適宜、OpenCVで定義されている関数を用いているが、その細部について調査できていない点には留意されたい。

*/

// インクルード---------------------------------------------------------------------------------------

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <cstdlib>
#include <string>

// インクルード終了------------------------------------------------------------------------------------



// 名前空間の定義--------------------------------------------------------------------------------------

using namespace std;
using namespace cv;

// 名前空間の定義終了-----------------------------------------------------------------------------------



// プロトタイプ宣言-------------------------------------------------------------------------------------

// 二値画像を生成する関数
void makeBinary(cv::Mat mat);

// 特徴量の正規化をする関数
vector<double> normalize(vector<double>);

// 特徴量の平均を求める関数
vector<double> calcAve(vector<double>, vector<double>, vector<double>, vector<double>);

// ユーザーが入力した画像に最も近い画像を「あ」から「お」で求める
char judgeChar(vector<double>);

// 近似率を取得する関数
vector<double> calcAppVal(vector<double>);

// ここに各位が作成した関数を追加
vector<double> calccon(vector<cv::Mat>);
std::vector<int> coordinate(cv::Mat);

// プロトタイプ宣言終了---------------------------------------------------------------------------------




// メイン
int main(int argc, char* argv[]) {
    /* 
        画像をコマンドライン引数から読み込み
        このとき、ユーザーから入力された画像が一番目、以降はあ段からお段までの順番で読み込む必要がある。
    */
    Mat test  = imread(argv[1],0);    // ユーザーから入力された画像
    Mat charA = imread(argv[2], 0);  // 「あ」の画像
    Mat charI = imread(argv[3], 0);  // 「い」の画像
    Mat charU = imread(argv[4], 0);  // 「う」の画像
    Mat charE = imread(argv[5], 0);  // 「え」の画像
    Mat charO = imread(argv[6], 0);  // 「お」の画像

    /*
        特徴量を格納する可変長配列
        
        concents : 濃度
        hRuns    : 水平方向ラン数
        vRuns    : 垂直方向ラン数
        ratios   : 外接矩形比
    */
    vector<Mat> binaryMats;                // 二値化した画像をすべて保持する動的配列
    vector<double> concents;               // 濃度
    vector<double> hRuns;                  // 水平方向ラン数
    vector<double> vRuns;                  // 垂直方向ラン数
    vector<double> ratios;                 // 外接矩形比
    vector<double> ave;                    // 正規化後の特徴量の平均を保持
    vector<double> appValFeatures;         // 「あ」から「お」に対する近似率を保持

    // サーバーに識別結果を返すための変数
    char result = 0;

    

// 処理の開始------------------------------------------------------------------------------------------------------------------------------------

    /*
        各画像を二値化する
    */
    makeBinary(test);
    makeBinary(charA);
    makeBinary(charI);
    makeBinary(charU);
    makeBinary(charE);
    makeBinary(charO);

    /*
        二値化した画像をvector<cv::Mat>型の動的配列に格納

        binaryNats : 二値化した画像をtest, あ, い, う, え, お　の順番で保持
    */
    binaryMats.push_back(test);
    binaryMats.push_back(charA);
    binaryMats.push_back(charI);
    binaryMats.push_back(charU);
    binaryMats.push_back(charE);
    binaryMats.push_back(charO);




    // ここに関数呼び出し処理---------------------------------------------------------
    
    concents = calccon(binaryMats);



    //--------------------------------------------------------------------------------
    

    /*
        vectorに格納されている特徴量を正規化
    */
    concents = normalize(concents);
   /* hRuns = normalize(hRuns);
    vRuns = normalize(vRuns);
    ratios = normalize(ratios);*/

    /*
        正規化された特徴量の平均を求める
    */
    ave = calcAve(concents, hRuns, vRuns, ratios);

    /*  
        特徴量を元にtestが「あ」から「お」のどれに近いかを判定し、
        その結果をresultに格納
    */
    result = judgeChar(ave);


    /*
        ユーザーの入力画像が「あ」から「お」にどれくらい近いかの近似率を計算する
    */
    appValFeatures = calcAppVal(ave);

    /*
        サーバーに識別結果を返す
    */
    printf("%d\n", result);

    /*
        近似率を返す
    */
    for (int i = 0; i < appValFeatures.size(); i++) {
        printf("%lf\n", appValFeatures[i]);
    }






// テスト用表示コード-----------------------------------------------------------------------------------------------
    /*namedWindow("test");
    namedWindow("か");
    namedWindow("き");
    namedWindow("く");
    namedWindow("け");
    namedWindow("こ");
    imshow("test",test);
    imshow("か", charA);
    imshow("き", charI);
    imshow("く", charU);
    imshow("け", charE);
    imshow("こ", charO);
    waitKey(0);*/
// ---------------------------------------------------------------------------------------------------------------------

    return 0;
} // メインここまで


// 以下、関数の実装----------------------------------------------------------------------------------------------------------
/*
    引数で与えた画像を二値画像にする関数
    画像は配列であるため、引数で与えた画像がそのまま二値画像に変換されることに留意されたい。

    この処理の中では、二値化のために画像内の画素値の最大値と最小値を求めている。
    (最大値 + 最小値) / 2
    を閾値とすることで、画像ごとに画素値の中央の値で二値化することができる。

    また、画像において、背景色よりも対象の文字色の方が明度が高い場合、二値化後の対象文字が白色文字になる問題が発生する。
    特徴量の一つである濃度は範囲内の黒画素の数を数えるため、背景色によって特徴量が変化することはあってはならない。
    そこで、二値化後に黒画素の数を数え、白画素よりも多い場合は背景色が黒になっていると判断し、白黒反転をする処理をしている。
    これは文字が画像の半分以上を占めないことを根拠としている。つまり、太い文字を画像領域一杯で撮影した画像などの場合は動作を保証できない。
*/
void makeBinary(cv::Mat mat) {
    unsigned char current = 0;   // 注目画素の画素値を格納
    unsigned char max = 0;       // 画像内の画素値の最大値
    unsigned char min = 255;     // 画像内の画素値の最小値
    long  cnt = 0;               // 二値化後の画像内の黒画素の数


    // 最大値、最小値の計算
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            /*
                注目画素の画素値が変数maxよりも大きいときに
                変数maxの中身を更新
            */
            if (mat.at<unsigned char>(y,x) > max) {
                max = mat.at<unsigned char>(y,x);
            }
            /*
                注目画素の画素値が変数minよりも小さいときに
                変数minの中身を更新
            */
            if (mat.at<unsigned char>(y,x) < min) {
                min = mat.at<unsigned char>(y,x);
            }
        }
    }


    // 二値化処理
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            current = mat.at<unsigned char>(y,x);
            // 閾値は画像内の画素値の中央の値
            if (current > ((max + min) / 2)) {
                current = 255;
            }else {
                current = 0;
            }
            mat.at<unsigned char>(y,x) = current;
        }
    }


    /* 
        黒画素が画像の過半数の場合に白黒反転
        変数cntの中身が正の値なら黒画素が多いという判断になり、
        画像を白黒反転させる
    */
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            if (mat.at<unsigned char>(y,x) == 0) {
                cnt++;
            }else {
                cnt--;
            }
        }
    }
    // ここで変数cntが正か負かを判定
    if (cnt > 0) {
        for (int y = 0; y < mat.rows; y++) {
            for (int x = 0; x < mat.cols; x++) {
                if (mat.at<unsigned char>(y,x) == 0) {
                    mat.at<unsigned char>(y,x) = 255;
                }else {
                    mat.at<unsigned char>(y,x) = 0;
                }
            }
        }
    }

    // ここまでの操作で画像配列matは二値画像に変換されている------------------------------------------------
}

/*

    vectorに格納されている4つの特徴量をそれぞれ正規化する

    vector<double> concents : 濃度
    vector<double> hRuns : 水平方向ラン数
    vector<double> vRuns : 垂直方向ラン数
    vector<double> ratios : 外接矩形比

*/
vector<double> normalize(vector<double> vec) {

    // 変数定義------------------------------------------------
    double ave = 0.0;      // 特徴量の平均
    double s = 0.0;        // 分散
    double stdev = 0.0;    // 標準偏差
    double current = 0.0;  // 正規化する際に一時的に値を代入
    //---------------------------------------------------------



    // 特徴量の平均を計算---------------------------------------
    for (int i = 0; i < vec.size(); i++) {
        ave += vec[i];
    }
    ave /= vec.size();
    //---------------------------------------------------------



    // 特徴量の分散を計算---------------------------------------
    for (int i = 0; i < vec.size(); i++) {
        s += (vec[i] - ave) * (vec[i] - ave);
    }
    s /= vec.size();
    //---------------------------------------------------------



    // 標準偏差の取得-------------------------------------------
    stdev = sqrt(s);
    //---------------------------------------------------------



    // 特徴量の正規化-------------------------------------------
    for (int i = 0; i < vec.size(); i++) {
        current = (vec[i] - ave) / stdev;
        vec[i] = current;
    }
    //----------------------------------------------------------
    
    return vec;
}

/*
    正規化された4つの特徴量を平均してvector<double>型の配列に格納して返す関数
    ここで、引数として受け取るvector<double>型の配列は、それぞれの正規化後の特徴量である

    concents : 濃度
    hRuns    : 水平方向ラン数
    vRuns    : 垂直方向ラン数
    ratios   : 外接矩形比
*/
vector<double> calcAve(vector<double> concents, vector<double> hRuns, vector<double> vRuns, vector<double> ratios) {

    vector<double> result; // 特徴量を平均した結果を保持するvector<double>型の動的配列

    /*
        ここで平均を計算
    */
    for (int i = 0; i < concents.size(); i++) {
        result[i] = (concents[i] + hRuns[i] + vRuns[i] + ratios[i]) / concents.size();
    }

    return result;
}

/*
    ユーザーが入力した画像が「あ」から「お」の、どの文字と近い特徴を持っているかを判断する関数

    vector<double> : 正規化された特徴量の平均
*/
char judgeChar(vector<double> vec) {

    double testFeature = 0.0;       // ユーザーが入力した画像の特徴量
    /*
        ユーザーが入力した画像の特徴量を変数testFeatureに代入
    */
    testFeature = vec[0];

    double min = 100000.0;          // もっとも1に近いものを算出するための変数
    char result = 0;                // 判別結果を一文字の数字として保持するための変数

    /*
        ここでもっとも近似率の高いものを判定する
        まったく同じ特徴量だった場合、計算結果は1となることは自明である
        この性質を使って判定をする

        識別結果の最大が1であるから、1にもっとも近い近似率の文字が取得したい結果である
        つまり、1 - 「あ」から「お」のそれぞれの近似率　を計算し、これが最も小さいものがもっとも近いひらがなであるということである

        そのときの i を結果として返すことで、関数の役割は終了する
    */
    for (int i = 1; i < vec.size(); i++) {
        if (1 - (vec[i] / testFeature) < min) {
            min = vec[i] / testFeature < min;
            result = i;
        }
    }
    return result;

}

/*
    ユーザーが入力した画像の特徴量が「あ」から「お」の特徴量にどれだけ近いかの割合を
    近似率としてvector<double>型の配列に格納する関数

    近似率 : 「あ」から「お」の特徴量 / ユーザーの入力画像の特徴量
*/
vector<double> calcAppVal(vector<double> vec) {

    double testFeature = 0.0;          // ユーザーが入力した画像の特徴量
    /*
        ユーザーが入力した画像の特徴量を変数testFeatureに代入
    */
    testFeature = vec[0];

    vector<double> result;             // 「あ」から「お」のそれぞれに対する近似率を格納するための動的配列

    /*
        近似率を求めて、返答用のvector<double>型の配列に格納
    */
    for (int i = 1; i < vec.size(); i++) {
        result[i - 1] = vec[i] / testFeature;
    }

    return result;
}

// 以下に各位が作成した関数の処理を追加

vector<double> calccon(vector<cv::Mat> mat){
   
    std::vector<int> point;
    vector<double> result;
    //point = coordinate();

    for (int i = 0; i < point.size(); i++) {
        std::cout << point[i] << " ";
    }
    std::cout << "\n";
    
    unsigned char s;
    for(size_t i = 0; i < mat.size(); i++){
        int con=0;
        point = coordinate(mat[i]);
        for(int y=point[0]+1; y < point[1]; y++){
            for(int x=point[2]+1; x < point[3]; x++){
                s = mat[i].at < unsigned char>(y, x);
                if( s == 0){
                    con++;
                }
	            mat[i].at < unsigned char>(y, x) = s;
	        }
        }
        result.push_back(con);
        return result;
    }
}

  /*  
    unsigned char s;
    for(int y=point[0]+1; y < point[1]; y++){
        for(int x=point[2]+1; x < point[3]; x++){
            s = mat.at < unsigned char>(y, x);
            if( s == 0){
                con++;
            }
	         mat.at < unsigned char>(y, x) = s;
	    }
    }
    for (int y = point[0]; y < point[1];y++){
        std::cout << (int)mat.at<unsigned char>(y,75) << std::endl;
    }
    return con;
}
*/
std::vector<int> coordinate(cv::Mat binImage){
    int up = 100000;
    int down = -100000;
    int left = 100000;
    int right = -100000;
    std::vector<int> point;
    unsigned char s;
    for(int y=0; y < binImage.rows; y++){
        for(int x=0; x < binImage.cols; x++){
            s = binImage.at < unsigned char>(y, x);
            if( s == 0/*黒*/){
                if(up > y) up = y;
                if(down < y) down = y;
                if(left > x) left = x;
                if(right < x) right = x;
            }
        }
    }
    point.push_back(up);
    point.push_back(down);
    point.push_back(left);
    point.push_back(right);

    return point;
}