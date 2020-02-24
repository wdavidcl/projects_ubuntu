// C program to generate random numbers 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
//Custom functions created
//int length(double array[]);
int inputs = 2;
int MAX_EPOCHS = 30;
int s = 4;
double input_a[4] = {0,0,1,1};
double input_b[4]  = {0,1,0,0};
double target[4]  = {1,0,0,1};
double beta = 0.5;
double min_error = 0.05;

// Driver program 
int main(void) { 
    // This program will create same sequence of  
    // random numbers on every program run  
    //printf("Length of parameter a : %d\n",length(input_a));

    srand(time(0));
    double w_old[2];
    for(int i = 0; i<inputs; i++) {
        w_old[i] = ((double) rand() / (RAND_MAX));
        printf("%lf \n",w_old[i]);  
    }

    double E;
    double delta_w [inputs];
    double y [inputs];
    double y_prima [inputs];
    double out;
    double errorTotal = 1;
    int epochs = 0;

    while (errorTotal>min_error && epochs<MAX_EPOCHS){
        errorTotal = 0;
        for(int i = 0; i<s;i++){
            y [i]= input_a[i]*w_old[0]+input_b[i]*w_old[1];
            E = target[i] - y[i];

            if (E>0) out=1;
            else out = 0;
                        
            printf("out= %f, target= %f, y(%f,%f)=%f, E=%f ****** ",out,target[i],input_a[i],input_b[i], y[i],E);
            errorTotal += 0.5*E*E;
            delta_w[0] = beta*E*input_a[i];
            delta_w[1] = beta*E*input_b[i];
            printf("w1: %f, w2: %f **** ",w_old[0],w_old[1]);
            // if (out!=target[i]){
            //     w_old[0]+=delta_w[0];
            //     w_old[1]+=delta_w[1];
            //     printf("cambiar pesos \n");
            // } else printf("no cambiar pesos \n ");
            w_old[0]+=delta_w[0];
            w_old[1]+=delta_w[1];
            printf(" \n ");
        }
        errorTotal/=4;
        epochs++;
        printf("******Error acumulado: %f, epocas: %i \n", errorTotal, epochs);
    }
    //printf("y: %f \n", y[i]);


    //END OF THE CODE
    return 0; 
} 

// int length(double array[]){
//     return (int)( sizeof(array) / sizeof(array[0]));
// }