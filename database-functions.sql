-- Database Functions for ScanMama API
-- Run these in Supabase SQL Editor

-- Function to upsert daily usage records
-- This is called from the API after each scan
CREATE OR REPLACE FUNCTION upsert_daily_usage(
    p_user_id UUID,
    p_date DATE,
    p_scan_count INTEGER,
    p_overage_count INTEGER
)
RETURNS void AS $$
BEGIN
    INSERT INTO usage_daily (user_id, date, scan_count, overage_scan_count)
    VALUES (p_user_id, p_date, p_scan_count, p_overage_count)
    ON CONFLICT (user_id, date)
    DO UPDATE SET
        scan_count = usage_daily.scan_count + p_scan_count,
        overage_scan_count = usage_daily.overage_scan_count + p_overage_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission to authenticated users
GRANT EXECUTE ON FUNCTION upsert_daily_usage(UUID, DATE, INTEGER, INTEGER) TO authenticated;
GRANT EXECUTE ON FUNCTION upsert_daily_usage(UUID, DATE, INTEGER, INTEGER) TO service_role;
