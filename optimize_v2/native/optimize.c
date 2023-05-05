#include <assert.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

float step_size = 0.1;
float throttle_fraction = 0.8;

static float calc_throttle_fraction(float step_size)
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

float next_throttle_fraction(int length, float const *const observed_rtt_list, float const *const target_rtt_list)
{
    int i;

    // if all observed RTT are smaller than target RTT, then decrease control
    for (i = 0; i < length; i++)
    {
        if (observed_rtt_list[i] > target_rtt_list[i])
        {
            step_size = MIN(step_size, -step_size / 2);
            goto out;
        }
    }

    // else, increase the control
    step_size = MAX(step_size, -step_size / 2);
out:
    printf("step_size: %f\n",step_size);
    calc_throttle_fraction(step_size);
    printf("throttle_fraction: %f\n",throttle_fraction);
    
    return throttle_fraction;
}

void fraction_to_throttle(float fraction, int length, float const *const sorted_mcs, float const *const sorted_thru, float *const out_sorted_throttle)
{
    int i;
    float link_fraction, normalized_thru;

    printf("length%d\n",length);
    link_fraction = 0;
    for (i = 0; i < length; i++)
    {
        printf("sorted_thru %f\n",sorted_thru[i]);
        link_fraction += sorted_thru[i] / sorted_mcs[i];
        printf("link_fraction %f\n",link_fraction);
    }
    normalized_thru = (link_fraction + fraction) / length;

    for (i = 0; i < length; i++)
    {
        out_sorted_throttle[i] = normalized_thru * sorted_mcs[i] - sorted_thru[i];
    }
    return;
}
