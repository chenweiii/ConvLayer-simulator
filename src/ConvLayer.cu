#include "ConvLayer.h"
#include "tensor.h"

static void feedforward(LayerBase *thiz)
{
    ConvLayer *conv = (ConvLayer *) thiz;
    /* prepare output on GPU memory */
    tensor_create(&conv->lb->output, conv->lb->batchSize, conv->lb->outputDim, conv->lb->outputDim, conv->lb->outputChannel);
    conv->lb->output->mallocDev(conv->lb->output);

    /* TODO: feedforward implementation */
    printf("Doing feedforward\n");
}

void ConvLayer_init(ConvLayer **thiz, int batchSize, \
                    int inputDim, int inputChannel, \
                    int kernelDim, int kernelAmount, \
                    LayerBase *preLayer, LayerBase *nextLayer)
{
    (*thiz) = (ConvLayer *) malloc(sizeof(ConvLayer));
    if (!(*thiz)) {
        printf("ConvLayer.cu: No available Memory\n");
        exit(0);
    }

    (*thiz)->lb = (LayerBase *) malloc(sizeof(LayerBase));
    if (!(*thiz)->lb) {
        printf("ConvLayer.cu: No availablle Memory\n");
        exit(0);
    }

    /* LayerBase */
    LayerBase *base = (*thiz)->lb;
    base->batchSize = batchSize;
    base->inputDim = inputDim;
    /* Padding*/
    base->outputDim = inputDim - kernelDim + 1;
    base->inputChannel = inputChannel;
    base->outputChannel = kernelAmount;
    base->input = NULL;
    base->output = NULL;
    base->preLayer = preLayer;
    base->nextLayer = NULL;
    base->feedforward = feedforward;
    /* ConvLayer */
    (*thiz)->kernelDim = kernelDim;
    (*thiz)->kernelAmount = kernelAmount;
    /* TODO: Initialize Weights and bias */
    ConvLayer_weight_init(*thiz);
    ConvLayer_bias_init(*thiz);
}

void ConvLayer_weight_init(ConvLayer *thiz)
{
    tensor_create(&thiz->weight, thiz->kernelAmount, thiz->kernelDim, thiz->kernelDim, thiz->lb->inputChannel);
    tensor *tzr = thiz->weight;
    for (int i = 0; i < tzr->D0; i++) {
        for (int j = 0; j < tzr->D1; j++) {
            for (int k = 0; k < tzr->D2; k++) {
                for (int w = 0; w < tzr->D3; w++) {
                    tzr->set(tzr, i, j, k, w, 0.5);
                }
            }
        }
    }
    tzr->mallocDev(tzr);
    tzr->toGpu(tzr);
}

void ConvLayer_bias_init(ConvLayer *thiz)
{
    tensor_create(&thiz->bias, thiz->kernelAmount, 1, 1, 1);
    tensor *tzr = thiz->bias;
    for (int i = 0; i < tzr->D0; i++) {
        tzr->set(tzr, i, 0, 0, 0, 100);
    }
    tzr->mallocDev(tzr);
    tzr->toGpu(tzr);
}
