#ifndef ISPC_COMMON_H
#define ISPC_COMMON_H

#define DEBUG 1
#define MATRIX_SIZE 8

#endif

#ifndef __ISPC_STRUCT_matElement__
#define __ISPC_STRUCT_matElement__

struct matElement
{
    float value;
    int row, col;
};

#endif