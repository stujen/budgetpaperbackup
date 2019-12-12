CDF  �   
      lon       lat       time             CDI       <Climate Data Interface version ?? (http://mpimet.mpg.de/cdi)   Conventions       CF-1.4     history      .Thu Nov 03 17:59:54 2016: cdo -a -settaxis,1861-01-16,12:00,1mon -setcalendar,360days -seltimestep,3301/5600 tas_Amon_MPI-ESM-LR_piControl_r1i1p1_200001-284912_anom_fldmean.nc CMIP5_IV_43.nc
Tue Apr 12 19:13:24 2016: cdo -fldmean -ymonsub tas_Amon_MPI-ESM-LR_piControl_r1i1p1_200001-284912_short.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_200001-284912_ymonmean.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_200001-284912_anom_fldmean.nc
Tue Apr 12 14:07:42 2016: cdo -ymonmean -seldate,2000-01-01,2849-12-31 tas_Amon_MPI-ESM-LR_piControl_r1i1p1_185001-284912_all.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_200001-284912_ymonmean.nc
Tue Apr 12 11:57:57 2016: cdo mergetime tas_Amon_MPI-ESM-LR_piControl_r1i1p1_185001-203512.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_203601-213012.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_213101-223012.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_223101-233012.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_233101-249912.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_250001-269912.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_270001-284912.nc tas_Amon_MPI-ESM-LR_piControl_r1i1p1_185001-284912_all.nc
Model raw output postprocessing with modelling environment (IMDI) at DKRZ: URL: http://svn-mad.zmaw.de/svn/mad/Model/IMDI/trunk, REV: 3315 2011-06-28T03:57:59Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.     source        �MPI-ESM-LR 2011; URL: http://svn.zmaw.de/svn/cosmos/branches/releases/mpi-esm-cmip5/src/mod; atmosphere: ECHAM6 (REV: 4556), T63L47; land: JSBACH (REV: 4556); ocean: MPIOM (REV: 4556), GR15L40; sea ice: 4556; marine bgc: HAMOCC (REV: 4556);   institution       $Max Planck Institute for Meteorology   institute_id      MPI-M      experiment_id         	piControl      model_id      
MPI-ESM-LR     forcing       N/A    parent_experiment_id      N/A    parent_experiment_rip         N/A    branch_time                  contact       cmip5-mpi-esm@dkrz.de      comment       +started in year 3424 of tra0066 spinup run.    
references       �ECHAM6: n/a; JSBACH: Raddatz et al., 2007. Will the tropical land biosphere dominate the climate-carbon cycle feedback during the twenty first century? Climate Dynamics, 29, 565-574, doi 10.1007/s00382-007-0247-8;  MPIOM: Marsland et al., 2003. The Max-Planck-Institute global ocean/sea ice model with orthogonal curvilinear coordinates. Ocean Modelling, 5, 91-127;  HAMOCC: http://www.mpimet.mpg.de/fileadmin/models/MPIOM/HAMOCC5.1_TECHNICAL_REPORT.pdf;     initialization_method               physics_version             tracking_id       $2c817c36-0db8-46ac-b2f5-16e65b2cab1b   product       output     
experiment        pre-industrial control     	frequency         mon    creation_date         2011-06-24T18:09:06Z   
project_id        CMIP5      table_id      ;Table Amon (27 April 2011) a5a1c518f52ae340313ba0aada03f862    title         AMPI-ESM-LR model output prepared for CMIP5 pre-industrial control      parent_experiment         N/A    modeling_realm        atmos      realization             cmor_version      2.5.9      CDO       @Climate Data Operators version 1.7.0 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           0   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           8   time               standard_name         time   units         month as %Y%m.%f   calendar      360_day    axis      T           @   tas                       standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   cell_methods      time: mean     history       J2011-06-24T18:09:06Z altered by CMOR: Treated scalar dimension: 'height'.      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_MPI-ESM-LR_piControl_r0i0p0.nc areacella: areacella_fx_MPI-ESM-LR_piControl_r0i0p0.nc            H                A��    ���A��    =>�
A��    ;���A��    =��A��    ����A��    ���A��    ��&�A��    �.�<A��    �Z)�A��    �d�A��    ��b�A�    ��G�A��    ���iA��    ���qA��    �+�)A��    ��jA��    ���PA��    ���A��    ���A�    �8?WA�    �Y#�A�    ��b�A�    �m<�A�$    �D��A��    ��IVA��    ���A��    �pA�    �Xk�A�    ��nA�    ��ۭA�    �S�+A�$    ��]�A�,    ���A�4    �t��A�<    ���iA�D    �'�A�    �|�!A�    �|JwA�    ��W=A�$    �u��A�,    ���A�4    �ZA�<    �/�A�D    �a��A�L    ��A�T    ��KA�\    �{F$A�d    ���?A�,    �N�:A�4    �|�A�<    ��8]A�D    �/�-A�L    ����A�T    �9��A�\    ��kA�d    ��3�A�l    ����A�t    ��uA�|    �D�AĄ    ��5�A�L    �y�xA�T    ����A�\    ���jA�d    ����A�l    �'�A�t    ��s�A�|    �LjAǄ    �#Aǌ    � �\Aǔ    �b��Aǜ    ��AǤ    �
�PA�l    <�-�A�t    =Y�A�|    >5�!Aʄ    >6U�Aʌ    =!�Aʔ    =O�Aʜ    =���Aʤ    <���Aʬ    ��Aʴ    =���Aʼ    >w�A��    =�?�A͌    >T�EA͔    >c�A͜    >=�vAͤ    =��Aͬ    =�a�Aʹ    >:�Aͼ    =��A��    =^%A��    =|�A��    =�(�A��    <O9�A��    =���AЬ    ���Aд    = #�Aм    >_	A��    >�A��    =���A��    ;�A��    ��!�A��    =���A��    =��0A��    ;�-A��    =�].A�    =^�A��    >G�1A��    >)A��    >b��A��    <���A��    >�A��    =D�A��    <I��A�    �FQA�    =�gA�    =I��A�    =_-2A�$    =��A��    =��A��    >ŀA��    >m��A�    =uc*A�    =Z�A�    <��jA�    <+iA�$    <&]�A�,    =�{A�4    =�(�A�<    ><��A�D    ����A�    ���A�    �0,WA�    �?��A�$    ��,^A�,    ���TA�4    �*�GA�<    =�D�A�D    > 1A�L    =�O!A�T    =��8A�\    ��H5A�d    >��A�,    ���A�4    ��!A�<    > �}A�D    =~�A�L    �"��A�T    <�A�\    =E
A�d    �� NA�l    ��DA�t    �D��A�|    �]9A݄    �=��A�L    ��6A�T    ��A�\    �j8�A�d    �'-;A�l    ���A�t    �'*A�|    �#��A��    �ZdA��    �	�A��    �M��A��    ��5_A�    �$A�l    �[�A�t    �^��A�|    ��sA�    <�A�    ;��A�    <NOQA�    =c�vA�    ���mA�    =ݯ$A�    =B�A�    >F�A��    >�|�A�    =�3�A�    >=1A�    >_;_A�    >�UA�    =�f�A�    ���A�    ��V�A��    ����A��    ��kFA��    �^DA��    �#-�A��    �C�'A�    ���uA�    �P�A�    ��l�A��    �ςA��    ����A��    ��z�A��    �75�A��    �}<QA��    �42�A��    �U�A��    =~�aA�    �Fj�A��    �ڃ�A��    ���A��    =1�2A��    =ˉ�A��    =2ZA��    <)#A��    ��n+A�    >%pA�    >HڮA�    >U1KA�    >��?A�$    >���A��    >��WA��    >�W�A��    >�gA�    >�IIA�    >��!A�    >��UA�    >��4A�$    >�UiA�,    >��qA�4    >���A�<    >��6A�D    >���A�    >�\�A�    <�EA�    =SmA�$    ��2�A�,    <�&?A�4    =���A�<    =8�]A�D    �g�rA�L    =+h�A�T    <T<}A�\    ��"�A�d    >)B�A�,    ="8�A�4    >�A�<    >+�A�D    =��A�L    ��/ A�T    ���A�\    �3GA�d    ��A�l    ��A�t    �nBJA�|    ����A��    ���A�L    �(ǇA�T    ����A�\    ��·A�d    �rA�l    � o�A�t    �P�ZA�|    <|+A��    =>sSA��    = OA��    ;��A��    =�B�A��    ���iA�l    ���A�t    �o��A�|    =���A��    >f��A��    =�^�A��    =ǣ�A��    =�@)A��    =e�^A��    <���A��    �:��A��    ��2A��    <�F@A��    �$ A��    =�A��    �
?�A��    �K{%A��    :��
A��    ��A��    =��A��    =�UA��    ���A��    �*��A��    �
{A��    �dD�A�    �Ee/A�    �=�A�    ���=A�    ���A�    �2>A�    ���A�    �$�A�    ��HtA�    �*�A�    <�L�A�    =5S�A    =q� A�    �I�A�    =��lA�    ��	$A�    ='�A�    �,yA�    :���A�    ���eA    �6��A    ��m�A    =��A    =�aA$    �C�A�    ;�EeA�    ���A�    �#i�A	    ���A	    ���rA	    ���,A	    ���
A	$    ��)HA	,    �W�A	4    ��1'A	<    ��;�A	D    �j((A    ���UA    ����A    ���A$    ��'A,    ��ܪA4    ����A<    ��}�AD    �D�.AL    =�Z�AT    ;�-�A\    =�Ad    =�PA,    ���A4    ��fHA<    �,y�AD    ��V0AL    ����AT    <�=$A\    <��Ad    =˙�Al    �?v�At    =#��A|    �ުXA�    ��yAL    �$AT    �:�A\    ��<Ad    ���UAl    ��DvAt    �_4�A|    <A(A�    <�>5A�    <�~TA�    =�&�A�    >aO5A�    <�"�Al    >�At    >��A|    =AA�A�    ���[A�    =�UA�    >/-�A�    >u�A�    >L�JA�    >c.bA�    >��6A�    =���A�    >\�A�    >"�A�    =?�rA�    =ɅjA�    >!��A�    =�OA�    =��oA�    =��A�    ��i�A�    ;��~A�    ����A�    =��bA�    �{��A�    <T\oA�    ��}�A�    ��lA�    =Q�aA�    ���A�    ;��@A�    =���A�    =��LA�    ����A�    =367A�    >h�A    >r��A�    >{A�    >a�PA�    >sGA�    <�A�    >0	3A�    =�A�    =�]A    =��A    ��0@A    ���A    <��A$    :G��A!�    �
�*A!�    �ÖDA!�    <��^A"    �t�MA"    �K�A"    ��ȠA"    � �HA"$    <�QKA",    �7|�A"4    �@��A"<    ��V@A"D    ����A%    ���hA%    ���A%    ����A%$    ��B�A%,    ��^�A%4    ��q�A%<    =�x�A%D    =��-A%L    ��MJA%T    ;şUA%\    �1�%A%d    �g6�A(,    =���A(4    ���	A(<    �LgtA(D    �
�A(L    ��1�A(T    �т^A(\    �J(A(d    ���A(l    ��uA(t    ��#�A(|    �]�A(�    ���qA+L    =RA+T    =)�LA+\    �j��A+d    �|��A+l    �g��A+t    =t(FA+|    >+wEA+�    =��9A+�    >Z�yA+�    >OˁA+�    >�5A+�    >r�=A.l    >���A.t    <h0$A.|    =u�TA.�    =�~A.�    >q�bA.�    >�A.�    >=��A.�    >I�PA.�    >��A.�    >�h�A.�    >�y�A.�    >ۂ�A1�    >�U2A1�    >��+A1�    >��A1�    >#��A1�    =��A1�    �}�A1�    � }FA1�    ���A1�    �^-A1�    =|� A1�    <��hA1�    <�&�A4�    >%z�A4�    ���A4�    �C�sA4�    �sA4�    �6� A4�    �^A4�    �mBuA4�    ��f�A4�    ���A4�    =�e~A4�    =4a�A5    =�,A7�    =�E�A7�    =�,@A7�    =��A7�    �&?9A7�    <���A7�    =5LA7�    >b CA8    >�C�A8    >=�_A8    >V2^A8    =��A8$    =��A:�    �SіA:�    =dv�A:�    >7A;    ��A;    =׬A;    =d�{A;    =���A;$    >
`�A;,    �v�A;4    =	@2A;<    =�?A;D    ���A>    ���9A>    �055A>    �sA>$    ����A>,    =� A>4    �*A�A><    ��4�A>D    ���A>L    <�@	A>T    =R�7A>\    >zИA>d    =�xAA,    >A�AA4    >
cAA<    =�#wAAD    ��טAAL    =�JqAAT    =�aAA\    =�z AAd    =1�AAl    =G�AAt    =֓rAA|    �ѧ�AA�    �
H=ADL    =�r+ADT    >-09AD\    =�{ADd    >tf�ADl    >9k�ADt    >`k�AD|    =��AD�    =��AD�    � �}AD�    =���AD�    ��k7AD�    =�>7AGl    >P|eAGt    >3�AG|    =���AG�    ��CAG�    ��\AG�    =qAG�    >KVAG�    > /AG�    =�;�AG�    �^-zAG�    <-1�AG�    <���AJ�    =�^�AJ�    >@l�AJ�    =\ҋAJ�    <�PAJ�    �(N�AJ�    ���(AJ�    ���AJ�    �3TAJ�    ��W�AJ�    ��_�AJ�    ���AJ�    =6,�AM�    =���AM�    >W��AM�    =��RAM�    �K��AM�    =��AM�    ��q�AM�    ��U�AM�    �rAM�    �[ytAM�    �x�lAM�    ���AN    �5�3AP�    �(ݺAP�    =�`�AP�    =�EAP�    �p�AP�    �P}AP�    =U�_AP�    =�I�AQ    =���AQ    =g�>AQ    =��AQ    >��aAQ$    >���AS�    >0%�AS�    =�+:AS�    >(��AT    =�QAT    =W�AT    <&��AT    <��AT$    �W@AT,    �Z�FAT4    �D=AT<    ���ATD    �G��AW    �>AW    ��WAW    �j��AW$    ��J�AW,    �YY�AW4    ���3AW<    ���JAWD    ��AWL    ��AWT    �?�qAW\    �CƱAWd    �X��AZ,    ���`AZ4    ��5�AZ<    ���iAZD    ��+�AZL    �z�AZT    ��LyAZ\    ���AZd    ��
PAZl    �#��AZt    �	6�AZ|    �� �AZ�    ��h�A]L    ���A]T    >h^�A]\    >jA]d    <��A]l    =�M�A]t    =�oA]|    >9p�A]�    >A]�    =�>�A]�    �A]�    ��d�A]�    >�A`l    =[�kA`t    <*�A`|    ;F��A`�    ���QA`�    ��}A`�    =��5A`�    >"��A`�    =�7A`�    =�x�A`�    =��2A`�    �޷(A`�    >T(Ac�    ���Ac�    ��߽Ac�    �8f$Ac�    =V �Ac�    =H�>Ac�    ��]Ac�    ����Ac�    � �sAc�    �e!�Ac�    ��I�Ac�    �i�Ac�    ��zAf�    >!3�Af�    ��W�Af�    �:r�Af�    ����Af�    �b�Af�    �B�}Af�    ����Af�    �v״Af�    =�Af�    ����Af�    =g��Ag    >{d2Ai�    <݆�Ai�    ���Ai�    �0��Ai�    �!Ai�    ���>Ai�    ��(�Ai�    =<={Aj    <���Aj    �>�5Aj    ��$�Aj    �k�SAj$    =�SAl�    =z{eAl�    ���Al�    >��Am    <ĴAm    ��_�Am    �4UbAm    ���HAm$    �TE�Am,    ��+BAm4    �X��Am<    �S��AmD    <��Ap    �<M<Ap    ����Ap    �r�$Ap$    �;F�Ap,    �}�\Ap4    ��bAp<    ��tApD    �	6�ApL    �/l2ApT    �U��Ap\    �{o�Apd    �\�uAs,    �R�9As4    �^��As<    ���AsD    ���AsL    ���tAsT    ���As\    � �Asd    ��Asl    ��xAst    �+�kAs|    ��%
As�    �Io�AvL    �X3�AvT    �C��Av\    ����Avd    ���Avl    ����Avt    ����Av|    ���Av�    ��d�Av�    �~��Av�    ��zrAv�    ��'AAv�    ��Ayl    ��D�Ayt    ��L-Ay|    �(t Ay�    �<{EAy�    �v�VAy�    ���mAy�    �=j�Ay�    �9F�Ay�    ���Ay�    ;q��Ay�    �7�Ay�    �u��A|�    ��U%A|�    ����A|�    ��A|�    ���A|�    ��NtA|�    �}6vA|�    �hQA|�    ��iBA|�    �A0]A|�    ��"0A|�    ��:�A|�    �:A�A�    ��:3A�    �/�NA�    �T��A�    ���A�    ��ڽA�    =/��A�    <��CA�    ���hA�    =�]	A�    =�YgA�    ��oBA�    <^
�A��    >/w�A��    >3pA��    >e"�A��    >��RA��    >��2A��    >�3�A��    >ٔJA�    >�.oA�    >��aA�    >�5�A�    >���A�$    >��A��    >�F�A��    >�]�A��    >��A�    >��A�    >Xo�A�    >\c�A�    >��A�$    >#��A�,    =T��A�4    >f!A�<    >[�lA�D    >�yA�    ��˸A�    >) A�    =+�A�$    ��A�,    ���uA�4    ��mA�<    �*rA�D    ��A�L    ��A�T    ��<�A�\    <]�&A�d    �+y�A�,    ���$A�4    ��=vA�<    �j �A�D    =�5�A�L    ��07A�T    �A�\    �>��A�d    �w��A�l    ��V�A�t    ���A�|    ��J�A��    ���=A�L    �;�A�T    :��A�\    ��+QA�d    ��KA�l    <b�A�t    ��ĻA�|    =���A��    �{��A��    <�"�A��    ���gA��    =�jfA��    =Vp�A�l    >�`�A�t    >&k"A�|    ���A��    �2��A��    =�ͤA��    =.�1A��    >j�A��    >��A��    =m�gA��    �J]WA��    <��0A��    ��}BA��    �H؞A��    ���KA��    ���A��    >�zA��    <w�!A��    =�(A��    ��.�A��    ���A��    �l�A��    =k�A��    <�T�A��    ���A��    <�*vA��    ��)A��    =>��A��    <
d�A��    ��.�A��    �-ٕA��    ��t�A��    ��A��    �#exA��    �!fA��    �c+�A�    �Bb�A��    �K+A��    =��A��    ��A��    ����A��    =�A��    ;��A��    =+,"A�    =��)A�    =0�%A�    <�[0A�    �FqkA�$    �'|ZA��    <�_�A��    ��A��    9�Q�A�    =�4�A�    ><xA�    =�R�A�    9[�VA�$    ���SA�,    =|_�A�4    �vM�A�<    ���JA�D    �V}�A�    �lgA�    ����A�    <�;VA�$    �]A�,    �^`�A�4    ���A�<    ���VA�D    ��A�L    ����A�T    <( MA�\    �7NA�d    <?��A�,    ��T�A�4    =�%�A�<    >E�&A�D    <3A�L    =�ryA�T    =C��A�\    =|��A�d    ><A�l    >pwA�t    >.c�A�|    >U�EA��    >�n
A�L    <�A�T    >`*A�\    >EܴA�d    >ɛpA�l    >���A�t    >��A�|    =�BhA��    >*��A��    >�'PA��    >[~"A��    >�W�A��    >�9WA�l    >b&fA�t    =_!�A�|    <�8�A��    ='�A��    =�n�A��    �<�A��    <r�@A��    <`�#A��    =6�A��    > -�A��    =�!�A��    =�oVA��    =��A��    >�3A��    >g��A��    =�t�A��    >�PA��    =���A��    ��1�A��    ="��A��    ����A��    <Ei�A��    ���@A��    =%�A��    ���"A��    =�7�A��    �",6A��    ��3A��    ����A��    ��o�A��    <���A��    ��A��    =(�NA��    �l��A��    =���A�    �7��A��    �w��A��    =�A��    =��qA��    <��A��    �7�ZA��    �)[A��    <�mA�    =�EjA�    =n>�A�    ��C(A�    ����A�$    ��ɴA��    � �2A��    �%�3A��    ����A�    �U��A�    ��p�A�    �V�2A�    ;!�A�$    >�A�,    =��A�4    =�!A�<    <S�A�D    ;��lA�    ���A�    �)��A�    <�X�A�$    =P`A�,    =g�>A�4    >,h�A�<    >"A�D    =Bn�A�L    9�vA�T    <ԩ�A�\    �"t�A�d    ���A�,    ��;�A�4    ��h]A�<    ��[BA�D    =�j}A�L    ����A�T    � 1WA�\    ����A�d    �3]A�l    �S�@A�t    �,@�A�|    �M��A��    ��k�A�L    ��@�A�T    >OA�\    >_�A�d    =]rrA�l    <�ۦA�t    <���A�|    <� ^A��    =`I�A��    >%$=A��    =(WxA��    >*�gA��    >�A�l    >p]A�t    >:��A�|    =��AĄ    =��AČ    9u��AĔ    =�<bAĜ    <�*�AĤ    <�FAĬ    =��aAĴ    =��Aļ    =���A��    =֪Aǌ    �� Aǔ    �?�"Aǜ    �0�AǤ    =j�AǬ    :���AǴ    =�$WAǼ    >K�{A��    =�i7A��    =�X�A��    >K��A��    >���A��    >z$�Aʬ    =/bAʴ    >]�Aʼ    <��A��    >Y]A��    >��A��    >�A��    >�QA��    =�ҎA��    =��A��    >ODA��    >�?sA�    >W� A��    >��A��    >�_�A��    >AŷA��    >��4A��    >��xA��    >dc�A��    =�_8A�    >NiA�    >(VcA�    >��WA�    >��9A�$    >��BA��    >���A��    >��A��    ?}�A�    >���A�    >S�A�    =�{�A�    �=�A�$    �UF#A�,    ;�TA�4    =8Q�A�<    ���1A�D    �c�A�    ����A�    �-�vA�    ��A�A�$    �ČGA�,    �p>�A�4    �Yn(A�<    �>h�A�D    �8�A�L    �Q�A�T    �SA�\    ��YuA�d    �d�1A�,    =
��A�4    �� �A�<    ��˙A�D    �0��A�L    �u��A�T    ��BJA�\    ���A�d    �J��A�l    �,�0A�t    �1ɐA�|    �dKAׄ    �taA�L    ����A�T    � �A�\    ��ZA�d    �l��A�l    �9ׁA�t    ��,A�|    ��|Aڄ    �fz}Aڌ    �c��Aڔ    ��C Aڜ    ����Aڤ    �b��A�l    ��MA�t    ����A�|    �$�'A݄    ��f�A݌    <Cz�Aݔ    �6�Aݜ    �=}Aݤ    ��*
Aݬ    ��Q�Aݴ    ��Aݼ    �/BxA��    ���bA��    ��_�A��    �m>A��    �5��A�    �,��A�    ���\A�    ���fA�    �/U�A��    =SaA��    �
�PA��    �iT�A��    <�_A��    =	OA�    >�A�    ��IA�    >28�A��    =ݥ�A��    =��A��    =.sA��    =ǭbA��    >fhA��    >ȀA��    >T�A��    >]�A�    >d��A��    =���A��    >=^pA��    >�?A��    =��/A��    �~��A��    ���A��    ��tA�    =�ͦA�    >_��A�    >_G�A�    =��A�$    =�ĽA��    =�\*A��    =�f�A��    ��aXA�    ����A�    ��;_A�    �||�A�    =�oA�$    =���A�,    <ՁA�4    =�GyA�<    >A�A�D    >���A�    >�A�    >��PA�    >� �A�$    >��A�,    >ݝUA�4    >�,A�<    >�":A�D    >�ӮA�L    >��A�T    ?	.A�\    ?:�A�d    ?':�A�,    ?+��A�4    ?$dcA�<    >�Q�A�D    ?�A�L    ?��A�T    >�LA�\    ?�A�d    >��EA�l    >�r9A�t    >侁A�|    >�@�A��    ?I�A�L    ?��A�T    >�PbA�\    >�A�d    >]��A�l    >>�rA�t    =��QA�|    =n�A�    <�X?A�    ���A�    ���1A�    =e~�A�    ��R�A�l    >g�7A�t    ��lA�|    =�R�A��    =���A��    ��ׄA��    �kH�A��    ��2A��    �ӭ�A��    ��Q�A��    ��qA��    �s�,A��    ��֔A��    �)�3A��    ���A��    � �[A��    ���A��    ���A��    ��0A��    �=�ZA��    ��F�A��    ����A��    ��(�A��    �}*�A��    �I��A��    �Eq�A��    ����A��    ��\�A��    �*�NA��    =\��A��    =P��A��    ���<A��    ��&�A��    =�.A��    ��q�A��    �{BA�    �r��A��    :���A��    ��A��    �G�oA��    �#X�A��    ���A��    ��;FA��    �7��A     ��`�A     ���\A     ����A     ��A $    �}.BA�    ��G�A�    �NiUA�    ���A    ��*A    �!�#A    �:�A    �Q%A$    ��T�A,    ����A4    ��A<    �2�AD    =[�$A    �2{�A    ����A    ��wA$    ���mA,    �YF�A4    �vI�A<    �\��AD    <K~AL    ;��$AT    �z�xA\    �P��Ad    ���A	,    ���A	4    �`�
A	<    ����A	D    =C��A	L    �)�A	T    �`�/A	\    ��*_A	d    ��.MA	l    <�C|A	t    <�4�A	|    =��*A	�    >XAL    =���AT    =��A\    �3{Ad    >9/�Al    >iqAt    >Pq�A|    >�A�    >� A�    >��A�    >K��A�    >N�A�    ><�Al    >�At    >�qA|    =�ӫA�    ���A�    ��uA�    =�3lA�    =��A�    >S�fA�    =���A�    =ZC�A�    >-A�    >�f�A�    >"��A�    =dA�    >��A�    = �A�    =��dA�    =l��A�    ��y�A�    =i��A�    ��j�A�    =��A�    >>��A�    =�G�A�    ���0A�    =���A�    �n�A�    �BZ�A�    �QAA�    <6~0A�    ��/ZA�    ���A�    ���A�    �7��A�    ��% A    ��VA�    ���A�    =���A�    ;�ӶA�    =��UA�    =���A�    =6�A�    =�mA    <�3�A    >��A    =�/A    �	A$    >��]A�    > @A�    =&��A�    >=`�A    >J+�A    >G9KA    > �hA    ��)cA$    ��ƣA,    =��bA4    =�"#A<    =%�*AD    �E�CA    >A    >���A    >��pA$    =}��A,    =�xA4    =���A<    >r�]AD    >��AL    =�H�AT    �(HA\    <l�Ad    >f��A",    >`<bA"4    >:�TA"<    >@��A"D    >1NA"L    >>��A"T    >�lA"\    =<IA"d    ��F�A"l    =#�A"t    =WzA"|    >IzA"�    >7�A%L    =�KA%T    ><�4A%\    >^i�A%d    >��uA%l    >��A%t    =|C~A%|    <���A%�    =���A%�    =;�A%�    =`�A%�    =}�A%�    =8*A(l    ��SA(t    =�U�A(|    �g׾A(�    ���A(�    ��3A(�    �*�A(�    �?�]A(�    ��GLA(�    �ǎ�A(�    �S��A(�    ��8A(�    �-.�A+�    �ذ�A+�    �R�7A+�    ��$�A+�    �Z@�A+�    ��A+�    ��=>A+�    ��bA+�    ���A+�    ��>�A+�    ��i�A+�    <Ώ�A+�    �e�A.�    �4\�A.�    ��A.�    ��eA.�    � "A.�    =1��A.�    =w��A.�    =�^UA.�    <)��A.�    =��2A.�    =��A.�    =��OA/    ��A1�    =�k6A1�    =bVHA1�    =�hA1�    ��A1�    ��� A1�    ���A1�    �%��A2    ��,�A2    ���A2    ��A2    ;��rA2$    ��"A4�    >]0A4�    >�A4�    �	��A5    ����A5    ��nA5    =��A5    <!�$A5$    =�_2A5,    ���8A54    >7��A5<    >egA5D    =*3A8    =��GA8    ��ΑA8    >4�A8$    >�BA8,    >4"�A84    >C�A8<    >�^A8D    >�S�A8L    >c�CA8T    >��A8\    >k�,A8d    >U(�A;,    >��A;4    >;�+A;<    >a��A;D    >�FDA;L    >S�A;T    >J��A;\    =��2A;d    <��uA;l    <��A;t    =���A;|    <�E�A;�    =o��A>L    ><щA>T    >!J8A>\    =�)�A>d    ;�*�A>l    >*kUA>t    =�(4A>|    =�E�A>�    <A��A>�    =��\A>�    =�}�A>�    =��A>�    <���AAl    �F�AAt    <� �AA|    �6�AA�    �&x�AA�    =�0AA�    ��:AA�    �ufAA�    ����AA�    ���AA�    ��)�AA�    �c�|AA�    =K�AD�    =4�AD�    ��AD�    =���AD�    =�,AD�    =d?�AD�    <2}RAD�    =���AD�    =j�dAD�    =��>AD�    >
e AD�    >�Q|AD�    =a��AG�    =�8�AG�    >#�$AG�    >�{�AG�    >Kf#AG�    <��AG�    =�K�AG�    =X�ZAG�    �`�xAG�    �4��AG�    >�AG�    >a��AH    >��AJ�    >GWAJ�    >m�AJ�    =��AJ�    =�N�AJ�    >J�jAJ�    >�>AJ�    =*L�AK    =v��AK    =�ěAK    =�jAK    >k�AK$    >F�rAM�    ���AM�    �ĳ�AM�    =�`oAN    >JE_AN    >.�AN    >E�oAN    >`�CAN$    >D��AN,    >�AN4    >�AN<    >9��AND    =���AQ    =�k�AQ    =��AQ    =e�2AQ$    ��	�AQ,    <�PAQ4    =�ԷAQ<    <r(dAQD    ����AQL    �G�AQT    =��IAQ\    =h��AQd    ��iAT,    �E�AT4    �控AT<    ����ATD    �+�]ATL    �%��ATT    ���AT\    =�jATd    =�i�ATl    =O��ATt    ���AT|    <�)�AT�    =�H�AWL    <��AWT    �| GAW\    ���AWd    �ܒ:AWl    �D�IAWt    ����AW|    ���JAW�    ��]QAW�    <�?�AW�    <�fAW�    �bXKAW�    �Q�AZl    ����AZt    ��8�AZ|    ����AZ�    �`rAZ�    �j,�AZ�    ����AZ�    ����AZ�    �%mAZ�    ��ٓAZ�    �w)'AZ�    �g�eAZ�    ����A]�    ��p�A]�    ����A]�    ��|A]�    ��֤A]�    ��H�A]�    ����A]�    ��82A]�    ��rA]�    <�O/A]�    <���A]�    =�iA]�    ��K�A`�    �
/A`�    ��� A`�    �c�A`�    �]!�A`�    �)ѮA`�    ���RA`�    ��TA`�    �_��A`�    �lG�A`�    �8�WA`�    ���PAa    �/EAc�    ��
�Ac�    �x��Ac�    ��UAc�    �~�Ac�    <��Ac�    �b١Ac�    =QX�Ad    �'�Ad    ��Ad    �\�Ad    �/1�Ad$    �a#Af�    ���Af�    �_"Af�    ��~QAg    ��JAg    �Q�Ag    =ehvAg    =t!Ag$    =�HKAg,    =���Ag4    =#s�Ag<    ��_�AgD    �K�Aj    <�toAj    ��.�Aj    �F�XAj$    ��j�Aj,    ��qAj4    ��,�Aj<    ��޽AjD    ���AjL    �T�aAjT    =:�vAj\    ��N�Ajd    ���Am,    �!�Am4    �.�;Am<    ����AmD    �<j�AmL    ��AmT    ���Am\    ���Amd    =`a(Aml    �0�WAmt    �f�xAm|    <�9,Am�    ���ApL    ��ApT    =��=Ap\    =�Apd    ����Apl    ��Apt    =��Ap|    =�<"Ap�    >��Ap�    =��Ap�    =��KAp�    <���Ap�    =8�Asl    >X+�Ast    >:oAs|    <U�As�    >(�eAs�    =�S�As�    =S�As�    >�CAs�    <�zEAs�    =�q�As�    <�As�    ���:As�    ���qAv�    =F�nAv�    �XUCAv�    ��`oAv�    ���Av�    ���*Av�    ���Av�    ���BAv�    �_cmAv�    �{��Av�    ���Av�    ���Av�    ���oAy�    ��O�Ay�    ��~Ay�    �+�3Ay�    ��Y`Ay�    ����Ay�    �=�Ay�    �$$�Ay�    �h��Ay�    �`m.Ay�    ��QAy�    �t��Az    �;�0A|�    ���A|�    ���VA|�    ��ݻA|�    ;���A|�    <��cA|�    <��|A|�    =x��A}    =�`sA}    <ÔA}    =ʦ#A}    =7�A}$    �(A�    ����A�    =�xQA�    =�?�A�    =%�A�    ����A�    �awA�    <���A�$    >϶A�,    <��UA�4    ���A�<    ��|�A�D    �'X�A�    ��x�A�    �A3�A�    ��H]A�$    �.��A�,    �W�A�4    ���8A�<    ���FA�D    �/~~A�L    ���A�T    ��a�A�\    �ȏA�d    ��q;A�,    �,Q�A�4    ��hA�<    <2�nA�D    �@� A�L    �Q��A�T    ����A�\    �ⱛA�d    =�pA�l    =48A�t    ��x�A�|    �}p�A��    ��7A�L    ��.A�T    :�)�A�\    ���A�d    ��f�A�l    <�ՠA�t    ���A�|    �I�MA��    �tYA��    �>�A��    �@�:A��    <Ӕ�A��    ��A�l    >!�CA�t    �u(�A�|    ���iA��    �9#}A��    �j�BA��    �� cA��    ��A��    �K��A��    ��A��    �L��A��    �KF�A��    ���A��    :�&A��    � �A��    �-�A��    =��A��    =�)A��    >E�A��    >N��A��    =��A��    =��A��    >�6A��    <�}�A��    >Qj:A��    >E`A��    >C�kA��    >�f�A��    >4}A��    =[t^A��    =<*�A��    =�}A��    =)�A��    ��&�A��    ��nlA��    �P3LA�    ��'�A��    ��Y�A��    >F}�A��    =�OA��    �]7�A��    =R��A��    =͓�A��    <B��A�    =M�!A�    =94VA�    �O$fA�    =h�:A�$    =�fvA��    � y�A��    �L��A��    =��A�    �h"�A�    ��A�    <"{�A�    �ף�A�$    =~��A�,    =��'A�4    :a��A�<    ����A�D    ��f2A�    �0�eA�    ����A�    ���A�$    �;��A�,    ;��uA�4    =�;�A�<    ;��A�D    => 0A�L    <�A�T    �B�A�\    ����A�d    ��IDA�,    =��A�4    >d��A�<    �Z�A�D    =��A�L    =�zA�T    �q�A�\    �[j�A�d    =c+�A�l    =ٸA�t    >ySA�|    =�[A��    �$��A�L    =��A�T    ����A�\    ��59A�d    =�M�A�l    <�XA�t    <�x�A�|    =��CA��    <j��A��    >�A��    >��A��    > ��A��    >�e*A�l    >���A�t    >��[A�|    >���A��    >0�2A��    > A��    >� A��    >_��A��    >i�	A��    >2��A��    >���A��    >�:A��    >!ͣA��    ��6A��    >/q�A��    =���A��    >�'<A��    >oeA��    >8nA��    >JSA��    =�{�A��    =pS�A��    ���A��    ����A��    =qYA��    �CA��    ����A��    �^��A��    >eA��    =�@A��    <��A��    ��`XA��    �.zuA��    =X}�A��    �WV�A��    >�AA�    >!�@A��    =�� A��    �4͙A��    ��8A��    >-��A��    >QȤA��    ><�A��    >$T-A�    =֕PA�    ��'A�    =��,A�    =�S�A�$    �Z�)A��    �A^�A��    ����A��    :l�WA�    =�A�    <���A�    �-pA�    ��'A�$    ��VA�,    �略A�4    ��A�<    �5��A�D    �ƍ�A�    ��A�    �F��A�    ��A�$    �U�A�,    ��dA�4    ��?A�<    �m�A�D    �(!�A�L    ��tA�T    �&UA�\    ��WA�d    =��A�,    =�w�A�4    �Qp�A�<    =!A�D    �R��A�L    �N�A�T    =R�"A�\    =u��A�d    =xi�A�l    =��;A�t    =�A�|    >`sQA��    >��A�L    =P�A�T    =�qA�\    >d<A�d    =�9fA�l    >0)qA�t    >^�A�|    > �bA��    :�.A��    ��T�A��    =(sA��    =06A��    �B0�A�l    =�A�t    =���A�|    =«QA��    >�A��    =�QA��    �ґ�A��    ��~#A��    �˘/A��    ���YA��    ���A��    ��dA��    =X�`A��    �Y/6A��    �;3�A��    �aVaA��    ���A��    ���A��    ��[A��    ��-A��    �6��A��    �	X�A��    ���ZA��    =L�mA��    �k��AĬ    >��AĴ    ���Aļ    =��0A��    =� 8A��    =.YGA��    ;C��A��    <��QA��    = BA��    <�H�A��    < ��A��    ;��RA�    =��A��    ��mkA��    ��A��    �z��A��    ��}A��    �X�A��    ��5mA��    ���TA�    �E�A�    �� �A�    �+�A�    =���A�$    >ZCA��    >V;SA��    >OCoA��    >@�A�    =�,�A�    >V��A�    >�n�A�    >8��A�$    >SzA�,    >@� A�4    >r-zA�<    =���A�D    >e9�A�    >.�pA�    ���KA�    �g�MA�$    ��KA�,    � A�4    ��tPA�<    ��+�A�D    ;�'A�L    �ĵAA�T    �k��A�\    �c�A�d    ���A�,    ����A�4    �\�jA�<    �V�OA�D    �&��A�L    �5�gA�T    �S}PA�\    �7�A�d    �gUBA�l    �-�A�t    �P�gA�|    ���CAф    ����A�L    �p�+A�T    ��A�\    �{<�A�d    ���A�l    ���A�t    ��Q�A�|    ��� AԄ    ��*AԌ    �G��AԔ    �_��AԜ    �Y|�AԤ    �5�6A�l    ��$�A�t    ��2�A�|    ��^Aׄ    <�#�A׌    =��yAה    <%�Aל    �h��Aפ    <�=�A׬    ;�guA״    =��A׼    =���A��    >i��Aڌ    >�Aڔ    �`��Aڜ    <A&�Aڤ    �C�mAڬ    �(L�Aڴ    �3Z Aڼ    �ߟWA��    <��GA��    ���A��    ��A��    <�a6A��    <M�Aݬ    �L3�Aݴ    �d��Aݼ    �H��A��    ���A��    �zM�A��    �}�A��    �%�$A��    �$�}A��    �@QYA��    ��V�A��    ���KA�    �WzA��    ���A��    ��QA��    ��5"A��    �3�A��    �N�vA��    �n6A��    �S�A�    �]�?A�    �{v�A�    �0
A�    <^iA�$    ���A��    �(�UA��    ��]A��    �qA�    ��k�A�    �H8�A�    �⇨A�    �@��A�$    ���A�,    �:eA�4    ���A�<    ��X�A�D    ����A�    =��JA�    ;i]�A�    �c�}A�$    �#�A�,    ��1A�4    ��A�<    ��A�D    �"�{A�L    ����A�T    ����A�\    =���A�d    =���A�,    ��#�A�4    >0`%A�<    =���A�D    ��A�L    ���A�T    <�;�A�\    ��Q�A�d    ����A�l    ���RA�t    =��.A�|    =�PIA�    ���JA�L    ��{�A�T    =̣CA�\    >GA�d    =R=*A�l    >,��A�t    >A�}A�|    >G�!A�    =��yA�    =�,TA�    >ZF�A�    >N{�A��    =�BxA�l    ��7�A�t    ���xA�|    <��8A��    =�A��    :�[RA�    ���A�    �
~�A�    ��֟A�    ����A�    ��A�    ;�L8A��    >+�%A�    �a�xA�    ��-A�    ��c�A�    �>��A�    �\�fA�    ��SA�    �&�YA��    ��(�A��    ���A��    =xA��    =���A��    =���A��    ="LA��    >0#�A��    >A��    =�YA��    <�k�A��    =�^�A��    ���A��    =��A��    ���}A��    �f��A��    �lNA�    ��XA��    �@P�A��    =�W�A��    �PܴA��    ��EA��    �#�6A��    �g��A��    ���0A�    �h��A�    ���#A�    ��-A�    �9/�A�$    ���A��    �5z%A��    �?��A��    ��h�A�    ��^VA�    �Q9A�    ��KA�    ����A�$    �RA�,    �;��A�4    �c
A�<    ��2�A�D    ���)A	     ����A	     =;ӺA	     ��2A	 $    =J+ZA	 ,    ���A	 4    �R�lA	 <    �E�A	 D    <ՉA	 L    <�O�A	 T    ���(A	 \    �4]�A	 d    ��A	,    <�G�A	4    �N�\A	<    �,��A	D    ��8A	L    ��kPA	T    ��R�A	\    ���A	d    ��`�A	l    ���A	t    ��GwA	|    ��,�A	�    ���8A	L    �E��A	T    ��B
A	\    �J;�A	d    ��љA	l    �^@A	t    ���A	|    �l�\A	�    ��2�A	�    ��A	�    �yA�A	�    ��iA	�    ��j)A		l    �`QA		t    ����A		|    =��A		�    �LhWA		�    =�aA		�    > �A		�    >;S�A		�    =�8�A		�    =�@A		�    =���A		�    >?�A		�    >��A	�    =���A	�    >+�A	�    =U�!A	�    >(��A	�    <��A	�    �uJA	�    <�P�A	�    �/e�