#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================

    
    // 获取数据指针
    const float* grad_ptr = static_cast<const float*>(grad->DataPtr());
    float* param_ptr = static_cast<float*>(param->DataPtr());
    float* m_ptr = static_cast<float*>(m->DataPtr());
    float* v_ptr = static_cast<float*>(v->DataPtr());

    // 遍历每个元素
    for (int64_t idx = 0; idx < grad->NumElements(); ++idx) {
        float g = grad_ptr[idx];    // 当前梯度

        // 1. 更新动量 m 
        m_ptr[idx] = beta1 * m_ptr[idx] + (1 - beta1) * g;

        // 2. 更新 v
        v_ptr[idx] = beta2 * v_ptr[idx] + (1 - beta2) * g * g;

        // 3. 偏差校正
        float m_hat = m_ptr[idx] / (1.0f - std::pow(beta1, t));
        float v_hat = v_ptr[idx] / (1.0f - std::pow(beta2, t));

        param_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
