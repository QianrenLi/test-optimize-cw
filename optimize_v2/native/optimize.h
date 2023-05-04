#ifndef __OPTIMIZE_H__
#define __OPTIMIZE_H_

float next_step_size(float, float const *, float const *);
float next_throttle_fraction(void);
void update_throttle(float, float const *, float const *, const float *);

#endif
