#include <assert.h>
#include <stdio.h>
#include "optimize.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) < (b)) ? (b) : (a))

// internal macros
#define WINDOW_SIZE 3
// internal constants
static const float DISCOUNT = 0.5;
static const float MIN_FRAC = 1E-4;
static const float ERR_PCNT = 0.10;
static const float INIT_STEP_SIZE = 0.1;

enum FLAGS {
    FLAG_STABLE = +1,
    FLAG_RESET  = -1,
    FLAG_NONE   = 0
};
typedef struct _sliding_window
{
    int ptr;
    int queue[WINDOW_SIZE];
} sliding_window_t;
static void sliding_window_append(sliding_window_t *window, int val)
{
    window->queue[ window->ptr ] = val;
    window->ptr = (window->ptr + 1) % WINDOW_SIZE;
}
static int sliding_window_check(sliding_window_t *window)
{
    int i;
    int _sum;
    int _flag;

    _sum = 0;
    for (i=0; i<WINDOW_SIZE; i++)
    {
        _sum += window->queue[i];
    }

    if      (_sum==WINDOW_SIZE)  { _flag = FLAG_STABLE; }
    else if (_sum==-WINDOW_SIZE) { _flag = FLAG_RESET; }
    else                         { _flag = FLAG_NONE; }
    return _flag;
}

// internal static variables
static float step_size = 0.0;
static float throttle_fraction = 0.0;
static sliding_window_t condition = { .ptr=0, .queue={0} };

/// @brief Coerce fraction between [MIN_FRAC, 1-MIN_FRAC].
/// @param step_size internal step_size
/// @return throttle fraction of files.
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

/// @brief Initialize the throttle fraction.
/// @param length length of the input arrays
/// @param sorted_mcs array of MCS of Links (or maximum link throughput)
/// @param sorted_thru array of average throughput of Links (excluding File)
/// @return initialized `throttle_fraction`
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
    step_size = INIT_STEP_SIZE;
    return throttle_fraction;
}

/// @brief Convert `throttle_fraction` to file throttle rate for each link.
/// @param fraction `throttle_fraction`
/// @param length length of the input arrays
/// @param sorted_mcs array of MCS of Links (or maximum link throughput)
/// @param sorted_thru array of average throughput of Links (excluding File)
/// @param out_sorted_throttle output array of file throttle rate for each link
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

/// @brief Main entry of the algorithm.
/// @param length length of the input arrays
/// @param observed_rtt_list observed RTT of each stream
/// @param target_rtt_list target RTT of each stream (constants)
/// @return `throttle_fraction`
float update_throttle_fraction(int length, float const *const observed_rtt_list, float const *const target_rtt_list)
{
    int i;
    int _flag;

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
    // append +1/-1 to sliding window
    _flag = FLAG_STABLE;
    for (i = 0; i < length; i++)
    {
        if (observed_rtt_list[i]>=target_rtt_list[i] || observed_rtt_list[i]<=target_rtt_list[i]*(1-ERR_PCNT) )
        {
            _flag = FLAG_RESET;
            break;
        }
    }
    // check sliding window continuity
    sliding_window_append(&condition, _flag);
    switch ( sliding_window_check(&condition) )
    {
        case FLAG_STABLE:
            return 0.0;
        case FLAG_RESET:
            step_size = INIT_STEP_SIZE;
            break;
        case FLAG_NONE:
            break;
    }

    calc_throttle_fraction(step_size);
    return throttle_fraction;
}
