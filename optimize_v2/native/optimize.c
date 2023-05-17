#include <assert.h>
#include <stdio.h>
#include "optimize.h"

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define IS_RISE(x) (x > 0)
#define IS_FALL(x) (x < 0)

// internal macros
#define WINDOW_SIZE 3
// internal constants
static const float DISCOUNT = 0.5;
static const float MIN_FRAC = 1E-4;
static const float ERR_PCNT = 0.10;
static const float STABLE_PCNT = 0.05;
static const float INIT_STEP_SIZE = 0.1;

enum FLAGS {
    FLAG_ANY_ABOVE   = +1,
    Z_FLAG_ANY_ABOVE = +WINDOW_SIZE,
    //
    FLAG_STABLE = -1,
    //
    FLAG_ALL_BELOW   = -2,
    Z_FLAG_ALL_BELOW = -WINDOW_SIZE * 2,
    //
    FLAG_OTHERWISE   =  0,
    Z_FLAG_OTHERWISE =  0,
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

    if      (_sum==Z_FLAG_ANY_ABOVE) { _flag = Z_FLAG_ANY_ABOVE; }
    else if (_sum==Z_FLAG_ALL_BELOW) { _flag = Z_FLAG_ALL_BELOW; }
    else                             { _flag = Z_FLAG_OTHERWISE; }
    return _flag;
}

// internal static variables
static float step_size = 0.0;
static float throttle_fraction = 0.0;
static sliding_window_t conditions = { .ptr=0, .queue={0} };

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

    // check and append current condition flag
    _flag = FLAG_ALL_BELOW;
    for (i = 0; i < length; i++)
    {
        if (observed_rtt_list[i] > target_rtt_list[i])
        {
            _flag = FLAG_ANY_ABOVE;
            break;
        }
        else if (observed_rtt_list[i] >= target_rtt_list[i]*(1-ERR_PCNT))
        {
            _flag = FLAG_OTHERWISE;
        }
    }
    if (_flag == FLAG_ALL_BELOW){
        for (i = 0; i < length; i++)
        {
            if (observed_rtt_list[i] > target_rtt_list[i] *(1-STABLE_PCNT))
            {
                _flag = FLAG_STABLE;
                break;
            }
        }
    }


    sliding_window_append(&conditions, _flag);
    // stable stopping
    if (_flag == FLAG_STABLE){
        step_size = 0.0;
    }
    else{
        // map from conditions to next step_size
        switch ( sliding_window_check(&conditions) )
        {
            case Z_FLAG_ANY_ABOVE:                                              // ABNORMAL: step_size should reset to negative
                step_size = -INIT_STEP_SIZE;
                break;
            case Z_FLAG_ALL_BELOW:                                              // ABNORMAL: step_size should reset to positive
                step_size = +INIT_STEP_SIZE;
                break;
            case Z_FLAG_OTHERWISE:                                              // OTHERWISE:
                if (_flag==FLAG_ANY_ABOVE)                                      //  step_size should > 0
                {
                    if (IS_RISE(step_size)) { step_size = -step_size / 2.0; }
                    break;
                }
                else                                                            //  step_size should < 0
                {
                    if (IS_FALL(step_size)) { step_size = -step_size / 2.0; }
                    break;
                }
        }
    }
    

    calc_throttle_fraction(step_size);
    return throttle_fraction;
}
