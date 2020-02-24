#include<Python.h>
#include<math.h>
//Global variable and definitions
double E = 2.71828182845904523536;
double T=0.1;                       //sample time in seconds

//Python exception method
static PyObject *ext_module_Error;

static PyObject *ext_module_print(PyObject* self, PyObject *args){
    const char* msg;
    int sts = 0;
    //check for the required arguments
    if(!PyArg_ParseTuple(args,"s",&msg)){
        //return NULL if nothing was found
        return NULL;        
    }
    if(strcmp(msg,"this_is_an_error")==0){
        //return NULL if the string is founded and set a text for the exception 
        PyErr_SetString(ext_module_Error,"This is a test exception");
        return NULL;
    }else{
        //print the message that was found
        printf("FROM PYTHON: %s\n",msg);
        sts=21;
    }
    return Py_BuildValue("i",sts);
}

static PyObject *ext_module_plant(PyObject* self, PyObject *args){
    double u;
    static double x;

    if(!PyArg_ParseTuple(args,"d",&u)){
        //return NULL if nothing was found
        return NULL;        
    }
    //calculate the behavior of G(s)=1/(s+1)
    x = (pow(E,-T))*x+(1-pow(E,-T))*u;
    //printf("FROM PYTHON x: %lf, ",x);
    return Py_BuildValue("d",x);
}

static PyMethodDef ext_module_methods[]={
    {"print",   ext_module_print,   METH_VARARGS,   "print a string from python"},
    {"plant",   ext_module_plant,   METH_VARARGS,   "calculate the behavior of a G(s)=1/(S+1)"},
    {NULL,      NULL,               0,              NULL}
};

static struct PyModuleDef ext_module =
{
    PyModuleDef_HEAD_INIT,
    "ext_module", /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    ext_module_methods
};

PyMODINIT_FUNC PyInit_ext_module(void)
{
    return PyModule_Create(&ext_module);
}