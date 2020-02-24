#include<iostream>
using namespace std;
#include<math.h>

double plant(double u);
double controller(double w, double y);
double E = 2.71828182845904523536;
double T = 0.1;               //sample time in seconds
double sim_time = 7;           //time in seconds
double reference = 7.83;

int main(){

    static double u,y;
    for (int i = 0; i < sim_time/T; i++)
    {
        /* code */
        u = controller(reference,y);
        printf("u: %lf, ",u);
        y=1*plant(u);
        printf("y(%lf): %lf\n",i*T,y);
    }
    return 0;
}

double plant(double u){

    static double x;

    x = (pow(E,-T))*x+(1-pow(E,-T))*u;
    printf("x: %lf, ",x);
    return x;
}

double controller(double w, double y){
    return -1.4048*(y)+2.4048*w;
}