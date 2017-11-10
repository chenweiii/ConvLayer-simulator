#include <stdio.h>
#include <stdlib.h>
#include "src/tensor.h"
#include "src/MemoryAllocator.h"
#include "src/net.h"

#define _INPUT_NUM 10

int main()
{
    MemoryAllocator_create(&ma);
    
    tensor *test;
    tensor_create(&test, _INPUT_NUM, _INPUT_DIM, _INPUT_DIM, _IMAGE_CHANNEL);
    
    /* Set input value */
    for (int i = 0; i < test->D0; i++) {
        for (int j = 0; j < test->D1; j++) {
            for (int k = 0; k < test->D2; k++) {
                for (int w = 0; w < test->D3; w++) {
                    test->set(test, i, j, k, w, 1.0);
                }
            }
        }
    }

    LayerBase *head = buildNetwork();
    tensor *result = trainNetwork(head, test);
    

    return 0;
}
