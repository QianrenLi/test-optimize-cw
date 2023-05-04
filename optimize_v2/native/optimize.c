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

float *update_throttle(float throttle, float *sorted_mcs, float *sorted_thru, float *out_sorted_throttle)
{
    int i;
    int length, _length, __length;
    float link_fraction, normalized_thru;

    length = (int)(sizeof(sorted_mcs) / sizeof(float *));
    _length = (int)(sizeof(sorted_thru) / sizeof(float *));
    __length = (int)(sizeof(out_sorted_throttle) / sizeof(float *));
    assert(length == _length);
    assert(length == __length);

    link_fraction = 0;
    for (i = 0; i < length; i++)
    {
        link_fraction += sorted_thru[i] / sorted_mcs[i];
    }
    normalized_thru = (link_fraction + throttle) / length;

    for (i = 0; i < length; i++)
    {
        out_sorted_throttle[i] = normalized_thru * sorted_mcs[i] - sorted_thru[i];
    }
    return out_sorted_throttle;
}
