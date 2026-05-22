-- stg_sales.sql
-- Staging model: cleans raw sales CSV, casts types, removes nulls and invalid rows.
-- Source: raw/synthetic_sales.csv loaded via dbt seed or external source

with source as (

    select * from {{ source('retail', 'synthetic_sales') }}

),

cleaned as (

    select
        cast(date as date)                          as sale_date,
        trim(upper(store_id))                       as store_id,
        trim(upper(product_id))                     as product_id,
        cast(quantity as integer)                   as quantity,
        cast(unit_price as numeric(10, 2))          as unit_price,
        cast(revenue as numeric(12, 2))             as revenue,
        cast(day_of_week as integer)                as day_of_week,
        cast(month as integer)                      as month,
        cast(week_of_year as integer)               as week_of_year

    from source

    where
        date is not null
        and store_id is not null
        and product_id is not null
        and quantity > 0
        and revenue > 0
        and unit_price > 0

)

select * from cleaned

