-- mart_weekly_sales.sql
-- Mart model: aggregates staging sales to store + week granularity.
-- Used as the primary input for Prophet and ARIMA forecasting models.

with stg as (

    select * from {{ ref('stg_sales') }}

),

weekly_agg as (

    select
        -- Week start date (Monday) as primary time dimension
        date_trunc('week', sale_date)::date         as week_start,

        store_id,

        -- Volume metrics
        sum(quantity)                               as total_quantity,
        count(*)                                    as transaction_count,
        count(distinct product_id)                  as unique_products,

        -- Revenue metrics
        round(sum(revenue), 2)                      as total_revenue,
        round(avg(unit_price), 2)                   as avg_unit_price,
        round(sum(revenue) / nullif(sum(quantity), 0), 2) as revenue_per_unit,

        -- Time dimensions (for BI slicing)
        extract(year from date_trunc('week', sale_date))    as year,
        extract(month from date_trunc('week', sale_date))   as month,
        extract(week from date_trunc('week', sale_date))    as week_of_year

    from stg

    group by
        date_trunc('week', sale_date),
        store_id

),

with_lag as (

    select
        *,
        lag(total_revenue, 1) over (
            partition by store_id order by week_start
        )                                           as prev_week_revenue,

        lag(total_revenue, 4) over (
            partition by store_id order by week_start
        )                                           as revenue_4w_ago,

        avg(total_revenue) over (
            partition by store_id
            order by week_start
            rows between 3 preceding and current row
        )                                           as rolling_4w_avg_revenue

    from weekly_agg

)

select
    week_start,
    store_id,
    total_quantity,
    transaction_count,
    unique_products,
    total_revenue,
    avg_unit_price,
    revenue_per_unit,
    year,
    month,
    week_of_year,
    prev_week_revenue,
    revenue_4w_ago,
    round(rolling_4w_avg_revenue, 2)                as rolling_4w_avg_revenue,

    -- Week-over-week growth
    round(
        case
            when prev_week_revenue > 0
            then (total_revenue - prev_week_revenue) / prev_week_revenue * 100
            else null
        end,
        2
    )                                               as wow_revenue_growth_pct

from with_lag

order by store_id, week_start
