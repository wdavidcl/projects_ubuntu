#include<stdio.h>
#include<math.h>

double plant(double u);
double controller(double w, double y);
double E = 2.71828182845904523536;
double T=0.1;               //sample time in seconds
double time = 7;           //time in seconds

void main(){

    static double u,y;
    for (int i = 0; i < time/T; i++)
    {
        /* code */
        u = controller(7.57,y);
        printf("u: %lf, ",u);
        y=1*plant(u);
        printf("y(%lf): %lf\n",i*T,y);
    }
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