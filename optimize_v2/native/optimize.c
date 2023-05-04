#include<assert.h>
#include<stdio.h>

#define MIN(a,b) (((a)<(b))?(a):(b))

static float step_size = 0.1;
static float throttle_fraction = 0.1;

float next_step_size(float *observed_rtt_list, float *target_rtt_list)
{
    int i;
    int length;
    int _length;
    
    length = (int)(sizeof(observed_rtt_list) / sizeof(float *));
    _length = (int)(sizeof(target_rtt_list) / sizeof(float *));
    assert( length == _length );

    // if exist any violation, continue with the direction
    for (i=0; i<length; i++)
    {
        if (observed_rtt_list[i] > target_rtt_list[i])
        {
            step_size = step_size; //FIXME:
            return step_size;
        }
    }

    // else, reverse the direction and half the step size
    step_size = -step_size/2; //FIXME:
    return step_size;
}

float next_throttle_fraction()
{
    if (throttle_fraction + step_size > 0)
    {
        throttle_fraction = MIN(0.9999, throttle_fraction + step_size);
    }
    else
    {
        throttle_fraction = 0.0001;
    }
    return throttle_fraction;
}

float *_update_throttle(float *sorted_mcs, float *sorted_thru, float throttle)
{
    
    return NULL;
}
