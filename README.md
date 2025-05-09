# LSTM-PINN-and-PINN-for-population-forecasting
# Abstract
Deep learning has emerged as a powerful tool in scientific modeling, particularly for complex dynamical systems; however, accurately capturing age-structured population dynamics under policy-driven fertility changes remains a significant challenge due to the lack of effective integration between domain knowledge and long-term temporal dependencies. To address this issue, we propose two physics-informed deep learning frameworks—PINN and LSTM-PINN—that incorporate policy-aware fertility functions into a transport-reaction partial differential equation to simulate population evolution from 2024 to 2054. The standard PINN model enforces the governing equation and boundary conditions via collocation-based training, enabling accurate learning of underlying population dynamics and ensuring stable convergence. Building on this, the LSTM-PINN framework integrates sequential memory mechanisms to effectively capture long-range dependencies in the age-time domain, achieving robust training performance across multiple loss components. Simulation results under three distinct fertility policy scenarios—the Three-child policy, the Universal two-child policy, and the Separate two-child policy—demonstrate the models’ ability to reflect policy-sensitive demographic shifts and highlight the effectiveness of integrating domain knowledge into data-driven forecasting. This study provides a novel and extensible framework for modeling age-structured population dynamics under policy interventions, offering valuable insights for data-informed demographic forecasting and long-term policy planning in the face of emerging population challenges.
# Problem Setup
See details in preprint: http://dx.doi.org/10.13140/RG.2.2.20232.12806

https://doi.org/10.48550/arXiv.2505.01819

DOI: 10.13140/RG.2.2.20232.12806
# PINN method
## Structure diagram
![PINN](https://github.com/user-attachments/assets/18e0c674-ff0a-46d5-861e-622836d03dc8)
## Resultes
![Figure_1](https://github.com/user-attachments/assets/59cd2bc3-1927-404f-852c-da089524e905)

![Figure_2](https://github.com/user-attachments/assets/5920d088-d9e8-488b-8d29-33408b72b7b4)

![Figure_3](https://github.com/user-attachments/assets/7eb47ac4-325b-4275-b979-737a778f2290)

![Figure_4](https://github.com/user-attachments/assets/354f2a2b-62f8-49b1-b133-e87faf4daa32)

![Figure_5](https://github.com/user-attachments/assets/571c53b9-d1d6-4286-b418-c7e77f9f3805)

![Figure_6](https://github.com/user-attachments/assets/45241146-110e-4d61-b3ae-1faedb15d9a8)
# LSTM-PINN method
## Structure diagram
![LSTM-PINN](https://github.com/user-attachments/assets/1b95c314-1e4f-477a-8145-ea689d8b83b6)
## Resultes
![Figure_1](https://github.com/user-attachments/assets/d93a9096-08b8-432f-b5b5-a43dd7fad22b)
![Figure_2](https://github.com/user-attachments/assets/d2e929d9-cc51-40d5-be59-71ebf4d62934)
![Figure_3](https://github.com/user-attachments/assets/cba9e5f2-46c9-4dd2-bdba-b4ea29dfdc16)
![Figure_4](https://github.com/user-attachments/assets/ef85185e-f425-4154-9d6c-35368306aa46)
![Figure_5](https://github.com/user-attachments/assets/408fbbfa-16b1-475c-a182-21c99eb5ffe5)
![Figure_6](https://github.com/user-attachments/assets/619c8e9b-4459-4bcb-a0ca-0ad5cd9e1777)
# Important Information
Although the assumptions in this work may not be fully complete and the code still requires multiple comparative experiments to select the optimal learning rate for both algorithms, we chose to release this work early. In doing so, we have successfully filled a gap in this field by introducing and demonstrating a brand-new algorithm. Moreover, I have open-sourced the code, so even if I don’t have the time to refine it further, others can continue to improve upon it. Most importantly, I managed to be the first to explore this direction by such a method — which is already a meaningful achievement.

