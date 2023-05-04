#include <assert.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

static float step_size = 0.1;
static float throttle_fraction = 0.1;

float next_step_size(float *observed_rtt_list, float *target_rtt_list, float his_step_size)
{
    int i;
    int length;
    int _length;

    length = (int)(sizeof(observed_rtt_list) / sizeof(float *));
    _length = (int)(sizeof(target_rtt_list) / sizeof(float *));
    assert(length == _length);

    // if all observed RTT are smaller than target RTT, then decrease control
    for (i = 0; i < length; i++)
    {
        if (observed_rtt_list[i] > target_rtt_list[i])
        {
            return MIN(his_step_size, -his_step_size / 2);
        }
    }

    // else, increase the control
    return MAX(his_step_size, -his_step_size / 2);
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
    int i;
    int length;

    float link_fraction = 0;
    length = (int)(sizeof(sorted_mcs) / sizeof(float *));
    for (i = 0; i < length; i++)
    {
        link_fraction += sorted_thru[i] / sorted_mcs[i];
    }
    link_fraction += throttle;

    float sorted_throttle[length];
    for (i = 0; i < length; i++)
    {
        sorted_throttle[i] = link_fraction * sorted_thru[i];
    }
    return sorted_throttle;
}
