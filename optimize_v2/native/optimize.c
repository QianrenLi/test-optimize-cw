#include <assert.h>
#include <stdio.h>
#include "optimize.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

static const float MIN_FRAC = 1E-4;
static const float DISCOUNT = 0.5;

static float step_size = 0.0;
static float throttle_fraction = 0.0;

static float calc_throttle_fraction(float step_size)
{
    if (throttle_fraction + step_size > 0)
    {
        throttle_fraction = MIN(1 - MIN_FRAC, throttle_fraction + step_size);
    }
    else
    {
        throttle_fraction = MIN_FRAC;
    }
    return throttle_fraction;
}

float update_throttle_fraction(int length, float const *const observed_rtt_list, float const *const target_rtt_list)
{
    int i;

    // if all observed RTT are smaller than target RTT, then decrease control
    for (i = 0; i < length; i++)
    {
        if (observed_rtt_list[i] > target_rtt_list[i])
        {
            step_size = MIN(step_size, -step_size / 1.1);
            goto out;
        }
    }

    // else, increase the control
    step_size = MAX(step_size, -step_size / 1.1);
out:
    calc_throttle_fraction(step_size);
    return throttle_fraction;
}

void fraction_to_throttle(float fraction, int length, float const *const sorted_mcs, float const *const sorted_thru, float *const out_sorted_throttle)
{
    int i;
    float link_fraction, normalized_thru;

    link_fraction = 0;
    for (i = 0; i < length; i++)
    {
        link_fraction += sorted_thru[i] / sorted_mcs[i];
    }
    normalized_thru = (link_fraction + fraction) / length;

    for (i = 0; i < length; i++)
    {
        out_sorted_throttle[i] = normalized_thru * sorted_mcs[i] - sorted_thru[i];
    }
    return;
}

float init_throttle_fraction(int length, float const *const sorted_mcs, float const *const sorted_thru)
{
    int i;
    float link_fraction;

    link_fraction = 0;
    for (i = 0; i < length; i++)
    {
        link_fraction += sorted_thru[i] / sorted_mcs[i];
    }
    link_fraction = 1 - link_fraction;

    throttle_fraction = link_fraction * DISCOUNT;
    step_size = 0.1;
    return throttle_fraction;
}
