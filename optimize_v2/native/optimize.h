#ifndef __OPTIMIZE_H__
#define __OPTIMIZE_H_

float next_step_size(float, float *, float *);
float next_throttle_fraction(void);
float *update_throttle(float, float*, float*, float *);

#endif
