--Crear base de datos de retirement_info solo con attrition

drop table if exists retirement_new;
create table retirement_new as
Select employeeid, attrition from retirement_info;

-- Unir base con retiros y manager
drop table if exists base_new4;
create table base_new4 AS
SELECT *
FROM manager_survey_data AS mgr
LEFT JOIN retirement_new AS rtn ON mgr.employeeid = rtn.employeeid;

--Crear base sin retiros
drop table if exists base_new5;
create table base_new5 as
Select employeeid, attrition,jobinvolvement,performancerating from base_new4;

--Crear base nueva
drop table if exists base_nueva;
create table base_nueva AS
SELECT general_data.*,
base_new5.attrition,
base_new5.jobinvolvement,
base_new5.performancerating,
employee_survey_data.EnvironmentSatisfaction,
employee_survey_data.WorkLifeBalance,
employee_survey_data.JobSatisfaction
FROM general_data
LEFT JOIN base_new5 ON general_data.employeeid = base_new5.employeeid
LEFT JOIN employee_survey_data ON general_data.employeeid = employee_survey_data.employeeid;



