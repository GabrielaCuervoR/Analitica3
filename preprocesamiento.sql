---Crear base de datos de retirement_info solo con attrition

drop table if exists retirement_new;
create table retirement_new as
Select employeeid, attrition from retirement_info;


--- Crear la base nueva con las 4 bases de datos juntas
drop table if exists base_nueva;
create table base_nueva AS
SELECT *
FROM general_data AS gd
LEFT JOIN manager_survey_data AS mgr ON gd.employeeid = mgr.employeeid
LEFT JOIN employee_survey_data AS emp ON gd.employeeid = emp.employeeid
LEFT JOIN retirement_new AS ret ON gd.employeeid = ret.employeeid;

