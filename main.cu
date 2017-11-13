#include <stdio.h>
#include <stdlib.h>
#include "src/tensor.h"
#include "src/MemoryAllocator.h"
#include "src/net.h"
#include "highgui/highgui_c.h"
#include "imgcodecs/imgcodecs_c.h"

#define FLOAT(x) (x * 100.0 / 100.0)

int main(int argc, char *argv[])
{
    MemoryAllocator_create(&ma);
    
    tensor *test;
    
    IplImage *img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!img) {
        printf("No availabe image\b");
        exit(0);
    }
    tensor_create(&test, 1, img->width, img->height, img->nChannels);

    /* Set input value */
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < test->D1; j++) {
            for (int k = 0; k < test->D2; k++) {
                uchar *ptr = (uchar *) img->imageData + j * img->widthStep;
                for (int w = 0; w < test->D3; w++) {
                    test->set(test, i, j, k, w, FLOAT(ptr[w]));
                }
                ptr += img->nChannels;
            }
        }
    }

    LayerBase *head = buildNetwork();
    tensor *result = trainNetwork(head, test);

    IplImage *output = cvCreateImage(cvSize(result->D1, result->D2), IPL_DEPTH_8U,  result->D3);
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < result->D1; j++) {
            for (int k = 0; k < result->D2; k++) {
                uchar *ptr = (uchar *) output->imageData + j * output->widthStep;
                for (int w = 0; w < result->D3; w++) {
                    ptr[w] = (uchar) ((int) result->get(result, i, j, k, w));
                }
                ptr += output->nChannels;
            }
        }
    }

    cvSaveImage("./output.jpg", output, NULL);

    

    return 0;
}
