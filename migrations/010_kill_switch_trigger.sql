-- DB trigger to hard-stop order inserts/updates if kill switch is on
CREATE OR REPLACE FUNCTION abort_if_kill_switch()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE ks bool;
BEGIN
  SELECT kill_switch INTO ks FROM execution_controls WHERE id=1;
  IF ks THEN
    RAISE EXCEPTION 'Kill switch active; refusing order insert/update';
  END IF;
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS trg_abort_if_kill_switch ON oms_orders;
CREATE TRIGGER trg_abort_if_kill_switch
BEFORE INSERT OR UPDATE ON oms_orders
FOR EACH ROW EXECUTE FUNCTION abort_if_kill_switch();