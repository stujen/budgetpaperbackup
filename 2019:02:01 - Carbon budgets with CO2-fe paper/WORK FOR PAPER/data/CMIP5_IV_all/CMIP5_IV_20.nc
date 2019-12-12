CDF  �   
      lon       lat       time             CDI       <Climate Data Interface version ?? (http://mpimet.mpg.de/cdi)   Conventions       CF-1.4     history      �Thu Nov 03 17:59:49 2016: cdo -a -settaxis,1861-01-16,12:00,1mon -setcalendar,360days -seltimestep,1313/3612 tas_Amon_CSIRO-Mk3-6-0_piControl_r1i1p1_020001-050012_anom_fldmean.nc CMIP5_IV_20.nc
Tue Apr 12 19:09:50 2016: cdo -fldmean -ymonsub tas_Amon_CSIRO-Mk3-6-0_piControl_r1i1p1_020001-050012_short.nc tas_Amon_CSIRO-Mk3-6-0_piControl_r1i1p1_020001-050012_ymonmean.nc tas_Amon_CSIRO-Mk3-6-0_piControl_r1i1p1_020001-050012_anom_fldmean.nc
Tue Apr 12 12:57:58 2016: cdo -ymonmean -seldate,200-01-01,500-12-31 tas_Amon_CSIRO-Mk3-6-0_piControl_r1i1p1_000101-050012_all.nc tas_Amon_CSIRO-Mk3-6-0_piControl_r1i1p1_020001-050012_ymonmean.nc
Minor changes were made to the model physics between years 1-160. Data for years 1-160 were subsequently discarded, and data for years 161-660 were remapped to years 1-500. 2011-05-11T07:25:34Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.   source        �CSIRO-Mk3-6-0 2010 atmosphere: AGCM v7.3.4 (T63 spectral, 1.875 degrees EW x approx. 1.875 degrees NS, 18 levels); ocean: GFDL MOM2.2 (1.875 degrees EW x approx. 0.9375 degrees NS, 31 levels)    institution       �Australian Commonwealth Scientific and Industrial Research Organization (CSIRO) Marine and Atmospheric Research (Melbourne, Australia) in collaboration with the Queensland Climate Change Centre of Excellence (QCCCE) (Brisbane, Australia)      institute_id      CSIRO-QCCCE    experiment_id         	piControl      model_id      CSIRO-Mk3-6-0      forcing       FN/A (Pre-industrial conditions with all forcings fixed at 1850 levels)     parent_experiment_id      N/A    parent_experiment_rip         N/A    branch_time                  contact      ^Project leaders: Stephen Jeffrey (Stephen.Jeffrey@qld.gov.au) & Leon Rotstayn (Leon.Rotstayn@csiro.au). Project team: Mark Collier (Mark.Collier@csiro.au: diagnostics & post-processing), Stacey Dravitzki (Stacey.Dravitzki@csiro.au: post-processing), Carlo Hamalainen (Carlo.Hamalainen@qld.gov.au: post-processing), Steve Jeffrey (Stephen.Jeffrey@qld.gov.au: modeling & post-processing), Chris Moeseneder (Chris.Moeseneder@csiro.au: post-processing), Leon Rotstayn (Leon.Rotstayn@csiro.au: modeling & atmos. physics), Jozef Syktus (Jozef.Syktus@qld.gov.au: model evaluation), Kenneth Wong (Kenneth.Wong@qld.gov.au: data management), Contributors: Martin Dix (Martin.Dix@csiro.au: tech. support), Hal Gordon (Hal.Gordon@csiro.au: atmos. dynamics), Eva Kowalczyk (Eva.Kowalczyk@csiro.au: land-surface), Siobhan O'Farrell (Siobhan.OFarrell@csiro.au: ocean & sea-ice)     comment       |Model output post-processed by the CSIRO-QCCCE CMIP5 Data post-processor for the IPCC Fifth Assessment. Dataset version: 1.0   
references       za) Rotstayn, L., Collier, M., Dix, M., Feng, Y., Gordon, H., O\'Farrell, S., Smith, I. and Syktus, J. 2010. Improved simulation of Australian climate and ENSO-related climate variability in a GCM with an interactive aerosol treatment. Int. J. Climatology, vol 30(7), pp1067-1088, DOI 10.1002/joc.1952 b) Please refer to online documentation at: http://cmip-pcmdi.llnl.gov/cmip5/     initialization_method               physics_version             tracking_id       $cffea417-8d05-47f8-87f2-ebea2fafbd2f   product       output     
experiment        pre-industrial control     	frequency         mon    creation_date         2011-05-11T07:25:34Z   
project_id        CMIP5      table_id      ;Table Amon (27 April 2011) 36bda60a55c2e1a01b1b1782daf91fea    title         DCSIRO-Mk3-6-0 model output prepared for CMIP5 pre-industrial control   parent_experiment         N/A    modeling_realm        atmos      realization             cmor_version      2.5.9      version_number        	v20110518      CDO       @Climate Data Operators version 1.7.0 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X              lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y              time               standard_name         time   units         month as %Y%m.%f   calendar      360_day    axis      T              tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   original_name         tsc    cell_methods      time: mean     history       J2011-05-11T07:25:34Z altered by CMOR: Treated scalar dimension: 'height'.      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_CSIRO-Mk3-6-0_piControl_r0i0p0.nc areacella: areacella_fx_CSIRO-Mk3-6-0_piControl_r0i0p0.nc                          A��    ���A��    =H]_A��    =UɈA��    =�t2A��    =�1�A��    =�:{A��    >��A��    ���nA��    =K�A��    >7�/A��    >�A�    >��A��    =P�0A��    ��"1A��    �/�A��    �-�A��    ����A��    =6�2A��    ���<A�    �q�A�    =y��A�    <\�A�    =�h-A�$    ����A��    <�|A��    >8A��    ��m�A�    �T��A�    �x�A�    �
�dA�    ���A�$    =�	�A�,    �!�A�4    ��a�A�<    >��A�D    =+-�A�    ����A�    =�	#A�    =kA�$    =^%�A�,    ��A�4    ��A�<    �(�*A�D    �7�YA�L    ��:A�T    �|��A�\    ���,A�d    <ڍA�,    ����A�4    ����A�<    �G��A�D    ���RA�L    �J/�A�T    �I\lA�\    ��$�A�d    ���TA�l    �s�!A�t    ��v�A�|    ��8AĄ    �5m�A�L    �� �A�T    <�U'A�\    =�x~A�d    =���A�l    =�sA�t    >���A�|    <�+�AǄ    =��cAǌ    =i+�Aǔ    =��Aǜ    =�`CAǤ    �)�<A�l    �&@�A�t    =C�A�|    =~Aʄ    �O��Aʌ    ��R�Aʔ    ���eAʜ    ��ϗAʤ    �ټ�Aʬ    �ħ�Aʴ    ��Aʼ    �$�UA��    �e<A͌    �)��A͔    �;eGA͜    ���}Aͤ    ����Aͬ    = ��Aʹ    ��#@Aͼ    ��?�A��    �z^A��    �[NjA��    ��A��    ��CA��    �z�AЬ    �[8�Aд    �[�tAм    �C�bA��    �a�A��    ��aA��    ��Q�A��    ��E�A��    ���/A��    ���A��    ��fRA��    ��bA�    ����A��    ��J�A��    ��pZA��    �	��A��    �0��A��    ���A��    ���A��    ���A�    ����A�    ��\A�    �f4LA�    ���A�$    �1�vA��    ����A��    �F}'A��    �?b�A�    ����A�    �2��A�    �y��A�    �a�A�$    ���mA�,    ���A�4    ���A�<    ��A�D    �6��A�    �<n�A�    ��FxA�    ���A�$    ��3A�,    ��AA�4    ��A�<    �%AsA�D    �gkA�L    ��2A�T    ��h�A�\    �A� A�d    �3;�A�,    �'��A�4    ��A�<    ��:dA�D    �Z�+A�L    ��z�A�T    ���0A�\    ��A�d    �&r�A�l    =1��A�t    ��;�A�|    ��p�A݄    ��9A�L    ����A�T    �KA�\    ��5�A�d    � ��A�l    �(��A�t    ���aA�|    �@�A��    ��Z�A��    �*`oA��    ��s:A��    =Jo�A�    =#� A�l    ��X�A�t    ��~MA�|    ��\DA�    =�eYA�    <+�A�    =���A�    >+p�A�    �[�A�    <)�eA�    ���A�    �0��A��    �_��A�    ���cA�    ��'�A�    �"oQA�    ��~A�    �#� A�    =��xA�    =�]#A��    >hA��    �J�PA��    ��v�A��    �<q�A��    =yA�    =��A�    <,�A�    >]�A��    >#�7A��    ��8�A��    �\c�A��    ��QA��    ��őA��    ��ЏA��    ��<�A��    �!7�A�    ���lA��    �
tA��    �1��A��    ���A��    �I�A��    ���4A��    <�LuA��    ����A�    �'��A�    ���A�    �t�A�    <�AA�$    �k��A��    =�7A��    �b�A��    <�5GA�    ; ��A�    <��A�    =��5A�    =(!�A�$    =��7A�,    :��oA�4    >@��A�<    >ɮA�D    �q��A�    =} A�    ��TrA�    ���A�$    =�8�A�,    =3��A�4    =l�A�<    �g1�A�D    >*7�A�L    =��A�T    <���A�\    ���A�d    ���#A�,    �{_�A�4    ��X�A�<    �ȴ4A�D    ��ݘA�L    �1oA�T    ���kA�\    �P��A�d    ��A�l    � �A�t    �a|A�|    �pDA��    <�)�A�L    ���A�T    =s9�A�\    =�]HA�d    =B_A�l    �/B�A�t    ��OA�|    ��VA��    ��� A��    >�TA��    ��'YA��    � ��A��    =�`HA�l    >i�A�t    ��b A�|    �AB�A��    <3.6A��    =�{IA��    =�˱A��    >u"�A��    >B�A��    ��:�A��    ����A��    =��_A��    ���A��    ��@�A��    ��A��    �%��A��    ��a�A��    �sb�A��    �ǻA��    <�Y�A��    <�3)A��    ��u�A��    ��yA��    <9�A��    ���OA�    =s�<A�    =�3�A�    ����A�    ���8A�    ���aA�    =<��A�    :�A�    >��A�    =��&A�    =�́A�    �s�_A    �G�_A�    �()�A�    �۞XA�    ���A�    �G�A�    �Z
�A�    =�A�    <�:�A    ��GA    =��mA    >8,A    ��lnA$    =1�A�    ����A�    =���A�    ;�A�A	    ���lA	    �a�gA	    �z��A	    �D�sA	$    =�aA	,    �a�A	4    <ȋ�A	<    �?�A	D    �
�A    ��A    ��� A    =��A$    ��N'A,    ����A4    �O��A<    ����AD    ��(�AL    �1�AT    =��JA\    >�6Ad    =[j�A,    =�e�A4    >DםA<    �f`6AD    =
SAL    =�D0AT    ��j�A\    >�5Ad    <v��Al    =��At    ����A|    =�]wA�    ��xAL    ���AT    =p$rA\    ;^�)Ad    =�Al    ���2At    ��]�A|    �Km"A�    �W��A�    ��j�A�    ��A�    ��>�A�    ��vAl    ���IAt    ����A|    �d�A�    �Π�A�    �2�;A�    �?lA�    ��0�A�    �!:A�    �`gA�    �x[>A�    �	YfA�    �"�"A�    �&�DA�    �	�vA�    �?��A�    ��mA�    �+OA�    �9�A�    ��A�    �8s%A�    �ǹA�    �R·A�    ��e�A�    ����A�    =�qA�    =�)-A�    <�Q�A�    <�U�A�    �@�A�    =�n�A�    >!�\A�    =�.PA�    >��+A�    =� kA�    =�A    �i�A�    �(�A�    �p�A�    =0A�    =s-�A�    =�/A�    �,�$A�    <��yA    ==��A    <��A    ��=�A    <e[�A$    =�w{A!�    >O��A!�    >ZIIA!�    >�TwA"    >�"A"    >Nm�A"    >C$kA"    =��FA"$    >��8A",    >�A"4    >��A"<    >TA"D    >���A%    >��gA%    >�AoA%    >�'�A%$    >�q�A%,    >��A%4    >�OtA%<    =�9�A%D    >��'A%L    >�4�A%T    >�VA%\    <d�A%d    =o�}A(,    =��&A(4    >A(<    >g�A(D    =�?�A(L    <�GA(T    =�A(\    >2"<A(d    >�8A(l    =��A(t    �+f�A(|    >y�ZA(�    >j�A+L    >z�A+T    =�.A+\    >>�3A+d    >W�1A+l    >3!A+t    >��A+|    =3��A+�    =��A+�    ;���A+�    �}�zA+�    ��,A+�    <�gA.l    �//JA.t    ����A.|    �)��A.�    �	2A.�    �� A.�    �@^|A.�    ���hA.�    �N�A.�    �,�A.�    �1��A.�    ��A.�    �)hPA1�    ��k�A1�    ��g�A1�    ���A1�    ����A1�    ��o
A1�    <C�?A1�    ��s]A1�    =�K�A1�    =�^�A1�    =��A1�    ;ڍA1�    ��]�A4�    ��A4�    ��d;A4�    ���4A4�    =n]�A4�    ���zA4�    �
NA4�    =;�A4�    �9uxA4�    <��A4�    <�?mA4�    >cA>A5    > M�A7�    > sA7�    =$��A7�    =�[A7�    =+M�A7�    >	��A7�    >I�A7�    >�A8    =�6�A8    >FA8    >\e	A8    :�% A8$    �pz�A:�    ���A:�    �� �A:�    ��L�A;    ���A;    <�A;    � ��A;    �COCA;$    =ҔA;,    <��A;4    <d�9A;<    �L�A;D    =��A>    ���vA>    ���wA>    =���A>$    >_mA>,    <�_@A>4    �S�A><    =�r�A>D    >8n�A>L    >��A>T    ���zA>\    =��nA>d    =�� AA,    =N��AA4    =�AA<    <���AAD    ���bAAL    ��	gAAT    �9ʎAA\    ���kAAd    ����AAl    �r�AAt    <�"AA|    ��%!AA�    ��y_ADL    ���<ADT    �e��AD\    =9X�ADd    �R˭ADl    =��ADt    �@}AD|    �7V>AD�    ���AD�    >
�>AD�    �#�AD�    ���QAD�    <�6�AGl    ;��nAGt    =���AG|    >�AG�    <31�AG�    =�u@AG�    ��CjAG�    �"�AG�    �rn�AG�    =��CAG�    >S�rAG�    =�aAG�    >'�oAJ�    =��`AJ�    =�nAJ�    >(��AJ�    =��AJ�    ����AJ�    =F�rAJ�    <���AJ�    =^qAJ�    ��AJ�    �F�AJ�    <��)AJ�    ����AM�    �OAM�    ���AM�    ����AM�    <i��AM�    �9X�AM�    �AM�    ��AM�    ��`�AM�    ��0AM�    �0�AM�    ����AN    ���AP�    �*�AP�    ��ZAP�    � �nAP�    ��GAP�    ����AP�    �b=�AP�    �z`�AQ    ����AQ    �]��AQ    �A�!AQ    ��E�AQ$    ��m�AS�    �d|�AS�    �y_�AS�    �֨�AT    ��AT    ���AT    ���6AT    �jAT$    �%�PAT,    �`&aAT4    ���AT<    ����ATD    ��5
AW    �D�$AW    �3ȈAW    ��`�AW$    ��!6AW,    ��w�AW4    �R�'AW<    �V8mAWD    �F��AWL    ���<AWT    ���0AW\    �__�AWd    ����AZ,    ���AZ4    ��RAZ<    ��n+AZD    �N&AZL    ���AZT    >z�AZ\    =��AZd    �`>AZl    <_$AZt    �4wAZ|    <�7	AZ�    <��A]L    =��3A]T    =~��A]\    �"+A]d    ����A]l    ��8�A]t    <.|�A]|    �@/�A]�    ���A]�    ��&�A]�    ;��JA]�    =�1A]�    =���A`l    <晪A`t    >(ǝA`|    >�HA`�    >~��A`�    =ۗA`�    �:ƹA`�    >2Z�A`�    <���A`�    >^��A`�    =��HA`�    �� �A`�    =���Ac�    =�ADAc�    =��`Ac�    �z
Ac�    ��o'Ac�    �eLAc�    ��l�Ac�    �RuqAc�    � ��Ac�    ���eAc�    �D;�Ac�    �V�Ac�    �Af�    �ulQAf�    �ne�Af�    ��lAf�    �4�Af�    �'��Af�    �^4PAf�    ����Af�    ��m�Af�    �ׂ�Af�    �YE�Af�    ��f�Ag    ����Ai�    �܋&Ai�    �OdAi�    =VݬAi�    ���Ai�    �zr�Ai�    ���LAi�    <��WAj    >A5lAj    >IP�Aj    =��Aj    >QbAj$    =��Al�    =��FAl�    >g'�Al�    >6�Am    >f{Am    >b&OAm    >��-Am    >�6.Am$    >�E�Am,    =�;Am4    >�H�Am<    ?�zAmD    >�ɽAp    >��Ap    >Ap    >$��Ap$    =EMAp,    ��mAp4    =�ڙAp<    =d��ApD    ���ApL    <�2lApT    =�l�Ap\    �j�Apd    ����As,    �b��As4    �ӵoAs<    <P��AsD    �V�AsL    =O��AsT    �	��As\    ����Asd    ��J�Asl    ��]Ast    �^O�As|    � ��As�    ��J�AvL    �m��AvT    ���Av\    ��@Avd    �Cn�Avl    �^f�Avt    �ќAv|    �T�Av�    ���Av�    ;k�Av�    <�tAv�    �h�+Av�    ���Ayl    �[4Ayt    ����Ay|    ;��Ay�    �D�Ay�    ����Ay�    �4<3Ay�    �M['Ay�    ��܌Ay�    �.Ay�    ��q_Ay�    =�9Ay�    =�i�A|�    =���A|�    =�u�A|�    =B�SA|�    =���A|�    >8"A|�    =´QA|�    =���A|�    =�D�A|�    >E+�A|�    =�},A|�    =�6�A|�    >�{,A�    >^�A�    >`��A�    >�X�A�    >{��A�    >j7A�    >LR�A�    =��A�    =��+A�    >1�A�    =��bA�    >y�_A�    >|�\A��    =�|�A��    >�UA��    >�A��    >''�A��    >IkDA��    =ɍ�A��    <F}3A�    =�5�A�    =JqfA�    9��zA�    �X'�A�$    ���A��    <�YA��    =o0�A��    �U�VA�    ��-�A�    ��ZA�    �l5A�    �U[�A�$    ��@>A�,    ���A�4    �jq�A�<    �0A�D    ����A�    ���MA�    <0�jA�    ��3A�$    �a�A�,    =���A�4    <�b,A�<    ���A�D    <a�A�L    ���XA�T    <��hA�\    ��'�A�d    =5�A�,    �*0A�4    �N�A�<    ����A�D    ��FA�L    ;��]A�T    ����A�\    �6�A�d    ���+A�l    ����A�t    �oaCA�|    ��yA��    ��XA�L    �CDA�T    <�$�A�\    �y�A�d    ��UA�l    <�cA�t    >W�A�|    =�B�A��    ���A��    =�.A��    =��XA��    =�FJA��    >T�A�l    =-A�t    >[9A�|    >6�wA��    >^�A��    >�$A��    =��A��    =�A��    >r_�A��    =eA��    >VBuA��    =�d�A��    >�=�A��    >\�A��    >"=A��    >m��A��    >I�OA��    =9{A��    <gZAA��    =���A��    >��A��    >�'�A��    ;NA��    >G�A��    =}<A��    <�3�A��    =L�gA��    >�"A��    ���A��    �aI�A��    ��БA��    >01A��    =�ƁA��    �;�cA��    ���qA��    >l�/A�    =�h3A��    >؁A��    =�u-A��    <��A��    <Ȼ'A��    <}�*A��    ���A��    =A��A�    >"HbA�    ���A�    ��S}A�    >H;DA�$    =�bA��    =�l�A��    >O�A��    �g\�A�    >)\oA�    ����A�    ;P�TA�    =V �A�$    >A�,    =�3A�4    =�ѪA�<    ��H�A�D    ���A�    =
�A�    ���{A�    =�WA�$    <.-hA�,    ��bA�4    <˅�A�<    >Q�A�D    <�hA�L    >j�A�T    =��-A�\    >/(A�d    >O��A�,    =@9�A�4    >.MA�<    >d��A�D    =֗�A�L    >ɢA�T    =��nA�\    >:��A�d    �BA�l    =�s�A�t    =��DA�|    =�o�A��    >)gA�L    >L��A�T    >N�|A�\    >2�	A�d    >.\�A�l    >���A�t    =�]�A�|    <�`A��    ;��A��    =���A��    >�c�A��    >&�A��    =���A�l    =��!A�t    =�bA�|    =�xA��    =� �A��    >7�A��    =��iA��    =���A��    <�n�A��    �&vSA��    >�˰A��    >���A��    =Q��A��    ;(DqA��    =�ϔA��    <��A��    �իA��    >���A��    >i�A��    >�t�A��    >�w�A��    >�HA��    >]�$A��    >�aA��    >Ī�A��    >�4�A��    >��A��    >܊�A��    >�h�A��    >ց:A��    >�e�A��    >�IA��    >���A��    >��1A��    >ݼ�A��    >�[aA�    >��)A��    >���A��    >�w�A��    >��&A��    >�6A��    >���A��    >S#�A��    >[�A�    >��A�    ��A�    �-�4A�    ��A�$    =h,9A��    =d�A��    =�řA��    ��hA�    =>��A�    ��׿A�    � gA�    �+D�A�$    �N0A�,    �2�A�4    ���A�<    �l-�A�D    ��A�    �)�[A�    ��A�    �h�A�$    ���A�,    =/�-A�4    �,͇A�<    �QA�D    �W��A�L    ��=GA�T    ���SA�\    ���IA�d    =�N�A�,    =��A�4    <͘A�<    ���6A�D    =�l	A�L    =g��A�T    =<M�A�\    =��A�d    >�A�l    >oA�t    >s��A�|    >�	A��    >�A�L    >�yA�T    >�U�A�\    >4��A�d    >A�pA�l    >g�A�t    >�i]A�|    >�?vA��    >���A��    =�CA��    =�A��    >M�A��    =\qA�l    <�Z~A�t    =��A�|    :;�AĄ    =`�4AČ    =�p�AĔ    =�[AĜ    <o/!AĤ    =��AĬ    > �AĴ    >��HAļ    =ɑ�A��    >��Aǌ    >�Aǔ    =M6Aǜ    ��=�AǤ    =��bAǬ    =�.AǴ    >.�&AǼ    >lJ�A��    >}�A��    >~�,A��    >���A��    >J�A��    ����Aʬ    =V�aAʴ    =��NAʼ    >8A��    =���A��    �?�A��    =�8;A��    =
�mA��    <���A��    =��A��    =zEwA��    =��A�    <�:,A��    ����A��    =�0UA��    =���A��    >$�A��    =��A��    =�>_A��    =��A�    >���A�    >v�A�    >{HA�    >4�XA�$    =�?�A��    >+j�A��    >(�A��    >
(
A�    >�A�    ��˘A�    > ǐA�    >9n�A�$    >�A�,    >M�
A�4    ��kVA�<    =WOA�D    =��A�    =(��A�    =f�3A�    �� �A�$    :x)�A�,    ��-�A�4    =MA�<    �w�BA�D    =M�A�L    >Q�A�T    >C_A�\    =��A�d    >��A�,    <�|xA�4    =��A�<    =ړ~A�D    =� A�L    >^�A�T    >U&�A�\    >��9A�d    >��A�l    >�}�A�t    >IP-A�|    >��Aׄ    =�vA�L    >j��A�T    >	/�A�\    >�A�d    >T��A�l    >8�A�t    <�1fA�|    =LdAڄ    �r�TAڌ    >���Aڔ    >C�Aڜ    >
�<Aڤ    =�dA�l    =���A�t    =��SA�|    =�epA݄    =԰A݌    =�kAݔ    =��NAݜ    <�'GAݤ    >!;�Aݬ    >#��Aݴ    <�@�Aݼ    �|_�A��    �}jA��    ���A��    ��%;A��    ��]A�    ���!A�    <�QA�    �Q�"A�    ����A��    <�A��    =��YA��    >HCA��    ��3�A��    ��,tA�    <Ǽ�A�    =Nn�A�    <���A��    =�V�A��    =��A��    �-�7A��    >�eA��    �mKA��    =B3`A��    >�WA��    =�N1A�    =n�A��    =π�A��    =Z�pA��    =�A��    =y.�A��    <�n A��    =�k�A��    >�:A�    >U�dA�    >P5A�    >-�A�    =�ΠA�$    =��A��    >�fA��    =�6�A��    >k)7A�    >e�A�    ��~�A�    =�@�A�    >8�vA�$    ��MtA�,    �SA�4    >�}�A�<    >`��A�D    >?{rA�    >[�A�    >D��A�    =,|A�$    =��A�,    =���A�4    >k�A�<    >�d�A�D    =���A�L    >��A�T    >��A�\    >��1A�d    >��<A�,    >Q)>A�4    >F$�A�<    =�MVA�D    =�U�A�L    <GDFA�T    ��@�A�\    >3��A�d    =r�YA�l    >
��A�t    >IO�A�|    >8�_A��    >={A�L    =l�}A�T    ;�A�\    <��7A�d    = ZsA�l    ��Z�A�t    =3�;A�|    =��A�    ��9A�    =x��A�    =fr#A�    ���A�    <��FA�l    �%JA�t    ����A�|    �O�A��    ���A��    ;��A��    =��A��    �Q�A��    >I�A��    =�WA��    =�ZA��    ��x7A��    ��SiA��    ���NA��    �~?�A��    �3I�A��    <4˦A��    =]��A��    =��A��    =�cKA��    ��VA��    ��(pA��    =e��A��    � ��A��    ��pTA��    ��oPA��    �=�HA��    ��C*A��    ����A��    ���)A��    �f$�A��    ���A��    �.N�A��    �G�A��    �,A��    �<GA�    ��,iA��    �r(iA��    ��6A��    ����A��    ���A��    ��n�A��    ���A��    ����A     �*&A     ��'A     �?7LA     ����A $    �&�A�    ��I�A�    ����A�    �g�A    ��2A    ��A�A    ����A    ;�^�A$    ��$�A,    ;b@�A4    �g�A<    =8!(AD    �
$A    �ONUA    =�/A    �w�sA$    �ݷ�A,    ��(A4    =F	pA<    =�QRAD    =�RAL    <���AT    >�A\    ��ӂAd    >Z�A	,    > ��A	4    =��A	<    >8�A	D    <.��A	L    >&�,A	T    =�݄A	\    >3"7A	d    � ��A	l    ���	A	t    =k�A	|    �t��A	�    �u0AL    =���AT    ={�\A\    ���Ad    ���Al    � T�At    <�A|    =�vFA�    =W�A�    > ��A�    =��!A�    =]U�A�    > ��Al    =K��At    ���tA|    �}��A�    ��eyA�    ��1�A�    <�KA�    =�53A�    �5R[A�    >��A�    >��A�    =��A�    ���sA�    =��A�    �lR�A�    >*'lA�    >�JA�    >KrA�    =vxA�    >rA�    =��A�    >���A�    >�.6A�    >b��A�    >�&A�    >�)(A�    >Y�A�    >��uA�    >$J�A�    ><i�A�    >��A�    >.6UA�    =��A�    <��A�    =��A�    =ڡ�A    =�MA�    =~iA�    =^�A�    =���A�    ><f�A�    >R��A�    =�
pA�    >$]<A    ;��A    =���A    �ébA    �U�A$    <�eqA�    =H�fA�    <�J�A�    <�j2A    �ƖA    =�)&A    =5/A    ���A$    �DiFA,    ��mDA4    �lA<    �'MAD    ���A    �j�,A    �%&�A    �2S�A$    �L��A,    ��=�A4    �PA<    ��6�AD    �?��AL    �S�MAT    �dI=A\    ��gAd    �a��A",    �LpEA"4    ��Z�A"<    �G��A"D    ��ÄA"L    � �A"T    ���A"\    =�%A"d    =:TA"l    =��A"t    �d� A"|    =jo�A"�    <v�'A%L    �ܵ�A%T    ;�1A%\    =�9�A%d    ����A%l    ���A%t    =���A%|    =�O�A%�    �'�A%�    ����A%�    >��A%�    =R �A%�    =�͛A(l    =�DYA(t    >�O=A(|    =ҹyA(�    >�A(�    =A�FA(�    �֏A(�    =W�A(�    ;���A(�    =��A(�    �ՠ�A(�    �W��A(�    ��nA+�    <�49A+�    �Q;A+�    ���A+�    ��A+�    ���]A+�    �^[A+�    ��>�A+�    ���A+�    �$�A+�    �냲A+�    <�'�A+�    =7��A.�    ����A.�    ��0]A.�    >w9A.�    >�WA.�    <��A.�    =��fA.�    �-[PA.�    =;��A.�    =���A.�    ��?�A.�    ="<A/    ��A1�    �M�A1�    �
��A1�    �7�A1�    =���A1�    ;�QXA1�    ;��A1�    =@zA2    >BF9A2    =�D=A2    >��A2    <ŤrA2$    =�u�A4�    =�C�A4�    =Y��A4�    <�?9A5    ��B%A5    =���A5    �#c�A5    ==A5$    �Ҭ%A5,    =wv�A54    ��"�A5<    ���TA5D    �ߣA8    �&C�A8    � :�A8    ���yA8$    ��^A8,    <�M@A84    � A8<    ��D�A8D    ����A8L    ���*A8T    �M�A8\    �N�A8d    �ҠA;,    �Z��A;4    ��wA;<    �Nn�A;D    ��G�A;L    <�9A;T    ���PA;\    <�O�A;d    ��#EA;l    ��L�A;t    ����A;|    �`�?A;�    ��϶A>L    �F�PA>T    =�TA>\    =9��A>d    =}ugA>l    >5�eA>t    =�G�A>|    =��"A>�    >�A>�    <�D�A>�    =�|A>�    =XM�A>�    >7��AAl    =ݐ�AAt    >?�+AA|    =�ώAA�    :x�AA�    >9�.AA�    =��7AA�    >F�AA�    >K{OAA�    ='QAA�    >H�AA�    �� �AA�    ;��0AD�    >2�AD�    >�/AD�    >�WAD�    > �AD�    =,�-AD�    =FmAD�    >�oAD�    >"PAD�    =���AD�    >&AD�    >x�AD�    >7ǓAG�    =��wAG�    �.��AG�    =�gpAG�    =0�AG�    =���AG�    ��a[AG�    ���pAG�    �$�yAG�    �@�AG�    ��(AG�    �:�@AH    ��AJ�    �TמAJ�    ����AJ�    ��[AJ�    �t(AJ�    ���AJ�    �q�AJ�    ��$�AK    ����AK    �TnAK    �G��AK    �]�AK$    ��%�AM�    �	*CAM�    ��lWAM�    ����AN    =A��AN    ����AN    ����AN    ���VAN$    �ԧsAN,    ��*�AN4    >Y.AN<    =$�=AND    ��v5AQ    �<lAQ    ��=nAQ    =|]\AQ$    >
�AQ,    =�CAQ4    =tAQ<    >KhAQD    =�9�AQL    =�u�AQT    =�PAQ\    =��pAQd    >D?AT,    >��>AT4    >ETAT<    >"�ATD    >X�FATL    >hK�ATT    >��
AT\    >�iiATd    >�$`ATl    >��)ATt    >K��AT|    >c��AT�    >%;�AWL    >GfAWT    >ƋAW\    =�YAWd    >)LyAWl    >��AWt    =�qAW|    =�wAW�    >|yAW�    >fj�AW�    =�uAW�    >{pvAW�    >ܝAZl    =�qAZt    <�ٴAZ|    =BErAZ�    =<
�AZ�    =���AZ�    =�AZ�    >!e@AZ�    �d!�AZ�    �(�UAZ�    �7��AZ�    �E�AZ�    �IrA]�    ��M\A]�    =PA]�    ��#�A]�    �/7�A]�    =���A]�    =�P&A]�    =�̇A]�    =�v�A]�    =�dlA]�    =�n�A]�    ��܁A]�    ��w�A`�    <t�&A`�    <_`�A`�    �זA`�    =��A`�    �K5,A`�    <xX�A`�    <}{A`�    <�
A`�    �YJbA`�    ���_A`�    ����Aa    �7�Ac�    �諄Ac�    ���Ac�    ���Ac�    ��;TAc�    �>�Ac�    ��_�Ac�    �/WjAd    ���WAd    ����Ad    �hj�Ad    ��վAd$    ��\Af�    ��+Af�    ����Af�    ��KAg    �aU�Ag    �p��Ag    �U8.Ag    �`�Ag$    ���CAg,    �jp�Ag4    ��->Ag<    �
�AgD    ���|Aj    �d��Aj    ����Aj    ��&�Aj$    �	}Aj,    ��k"Aj4    ���Aj<    =��AjD    ��_�AjL    =���AjT    ��2�Aj\    �a)QAjd    >Ry�Am,    >!;�Am4    :\۩Am<    =B��AmD    =׸�AmL    �8�AmT    �C�#Am\    �6��Amd    >$[Aml    =(YAmt    =���Am|    ��r3Am�    >'N�ApL    =��sApT    =XߑAp\    �BApd    =�WApl    =��>Apt    <`� Ap|    �,[eAp�    ��2Ap�    =�;�Ap�    <(��Ap�    ��<EAp�    �B�Asl    ��=�Ast    =�tAs|    �x��As�    =��As�    =q:As�    � 7tAs�    ��b�As�    <�As�    <���As�    ��,LAs�    =-V�As�    ��"Av�    �u��Av�    <��Av�    ;�s`Av�    �V�UAv�    :~<Av�    ��mAv�    =�F�Av�    =��PAv�    >�Av�    <QNAv�    =��{Av�    =�p�Ay�    =CۥAy�    = ��Ay�    >H�Ay�    ���OAy�    �Z�Ay�    �6��Ay�    �� ~Ay�    ��[Ay�    ��q�Ay�    �Uc�Ay�    ��
Az    �6�A|�    ����A|�    ���A|�    ���A|�    ��}tA|�    ��A|�    �kDA|�    ����A}    ���A}    ���A}    ��8�A}    ��V,A}$    ���BA�    <���A�    ;z��A�    =k�A�    =���A�    �4�A�    <��A�    =%FnA�$    >�A�,    >��tA�4    >$��A�<    <�9TA�D    =���A�    �}�A�    9��A�    ���A�$    ����A�,    >�S�A�4    <�rA�<    =9�A�D    >,x<A�L    =���A�T    >CJA�\    =�%fA�d    =�F;A�,    =�a�A�4    >oA�<    >�4A�D    >�\A�L    =ɂFA�T    >	6�A�\    >�ʶA�d    >f(/A�l    =�vA�t    >��A�|    =ȅnA��    ���:A�L    =�*]A�T    ��fA�\    ���(A�d    <���A�l    =�2A�t    <ĨiA�|    �/�A��    ��t�A��    �>%�A��    �_>1A��    ��J�A��    ��cA�l    �� �A�t    ��-�A�|    �?�A��    =�9A��    =�A��    <�B�A��    ;��A��    >,={A��    >2�?A��    >
�yA��    =���A��    =5=�A��    =� �A��    <� �A��    =�A��    =j��A��    <�hSA��    �aD�A��    ��ڸA��    �?��A��    �DA��    >/��A��    =L�!A��    �۞OA��    ���A��    �$w+A��    =b��A��    ��A��    =�l�A��    <�˸A��    =�*�A��    =�"6A��    �
)hA��    =I5qA��    =��A�    =��A��    >6�A��    ���A��    ���^A��    ���+A��    ��4!A��    �L(A��    ��O�A�    ;��A�    �ux�A�    >�OA�    =T�zA�$    ����A��    ����A��    ��C|A��    =��A�    ��y{A�    �T_�A�    ���gA�    �?FA�$    ��X�A�,    ��"�A�4    ��@�A�<    �=�A�D    ���[A�    �X��A�    �?:QA�    �~�A�$    ����A�,    ����A�4    ��1�A�<    �+�HA�D    �bؠA�L    ��s�A�T    ���A�\    �H*HA�d    ��n�A�,    �&c�A�4    ��|A�<    �2:A�D    �7�}A�L    �P�A�T    �*�lA�\    �ݠCA�d    ��;BA�l    <uqbA�t    <�jA�|    ���A��    ���\A�L    �X�FA�T    �2R�A�\    ���A�d    ���$A�l    ��-|A�t    ���jA�|    ����A��    �?A��    �O�KA��    �W��A��    ��<�A��    �Q�kA�l    ���A�t    ���0A�|    ���A��    ��>A��    �}�A��    <!PA��    =6?�A��    �Æ\A��    �E2�A��    �A��    =]A��    :N�A��    <:V	A��    =�r�A��    ��A��    ���A��    =��5A��    �#0A��    �L�{A��    ����A��    �%��A��    ��YA��    ��c[A��    �֯RA��    �-<A��    ��lA��    �4��A��    � �|A��    �@��A��    >&וA��    >H`A��    >*�A��    =P��A��    =|KA��    >$�.A�    =��$A��    <;��A��    ��fpA��    �}��A��    ��MA��    ��k�A��    <�jA��    =XԵA�    � [�A�    =�$A�    =���A�    >1E*A�$    =�m^A��    >JPA��    =��A��    >7�A�    >���A�    =� _A�    =�'�A�    >"�A�$    >��A�,    :�r�A�4    >
�qA�<    =J�A�D    >}rA�    <�v�A�    <�g;A�    =P��A�$    =�n�A�,    ��^�A�4    >?��A�<    >)2�A�D    >A�L    >j�8A�T    >(�'A�\    >c�~A�d    >y�:A�,    >��A�4    =Ѓ�A�<    =�-A�D    =�v�A�L    =�n�A�T    =���A�\    =i�A�d    <��ZA�l    �y�A�t    <�F5A�|    ��K�A��    ��A�L    =\#4A�T    =i�A�\    :�+_A�d    �$[�A�l    ��A�t    ��IA�|    ����A��    ���A��    �7�A��    ���pA��    ����A��    =���A�l    <ѺA�t    ����A�|    ��V�A��    ;�$zA��    ��P#A��    =���A��    �>LPA��    ����A��    ���A��    =�8_A��    =�K�A��    �(6�A��    =T�A��    =�anA��    5|�A��    ��A��    ��`7A��    ��
0A��    =k#AA��    =o�"A��    =��UA��    >��A��    =�b*A��    ��iAĬ    > ��AĴ    =�ޱAļ    �l�A��    =�A��    =�z�A��    =�LKA��    >�{A��    ;=B�A��    =�(A��    >\�A��    ����A�    =��,A��    =2MA��    =&�A��    :AԮA��    =G@�A��    =��1A��    �I�A��    �g�A�    ���sA�    =�eA�    �x��A�    �m&
A�$    ��	�A��    ��_A��    �7X�A��    ����A�    ��FAA�    �kA�    ����A�    ��S�A�$    �b��A�,    �  ]A�4    ���<A�<    ���|A�D    ���
A�    �X�#A�    �`.A�    �k�
A�$    ���A�,    ����A�4    �d�A�<    �l�A�D    ���fA�L    �ᲮA�T    �L<�A�\    ���A�d    �H�A�,    �U��A�4    ���A�<    ���A�D    ��A�L    ��&A�T    �&�A�\    �*�MA�d    ���A�l    ��MwA�t    ���A�|    �0��Aф    �4JQA�L    �r�A�T    ����A�\    �,��A�d    �C�A�l    ����A�t    ���MA�|    ��dAԄ    =�AԌ    =��<AԔ    >h['AԜ    >Z�|AԤ    �{�A�l    =��@A�t    =�s0A�|    =1��Aׄ    =Ao2A׌    <4ŧAה    ;��Aל    �ϥfAפ    ���A׬    ��۬A״    =˺�A׼    =޶A��    <�8�Aڌ    :��<Aڔ    ��kAڜ    ��}�Aڤ    ��WAڬ    ����Aڴ    ����Aڼ    ����A��    =٩A��    ="��A��    �(�qA��    <��cA��    �Y1Aݬ    ���UAݴ    �|�Aݼ    ���hA��    ���A��    <�P(A��    �	�A��    <��A��    �2��A��    ��ؠA��    ���A��    ��-A�    ���A��    ���~A��    �b�HA��    ��m�A��    ��)�A��    ��?WA��    ����A��    �A6MA�    ��<A�    ����A�    ��Y�A�    ��H�A�$    ��-DA��    ��Y�A��    ���A��    ��eA�    ����A�    �N�wA�    ���A�    �m'A�$    ���'A�,    �B�YA�4    ��ƮA�<    �E#A�D    ���A�    ;�A�    �,r�A�    �B�?A�$    ���A�,    <}�MA�4    =��A�<    ��b�A�D    �[�&A�L    �� 4A�T    =�`�A�\    ����A�d    �=bA�,    �v-�A�4    �9��A�<    ��5A�D    �T�A�L    �.�A�T    �<�A�\    ��nA�d    � <�A�l    ���rA�t    ��YA�|    �2ZA�    ����A�L    �^�DA�T    �p��A�\    �KA�d    �^��A�l    ��l�A�t    < :A�|    �m�A�    �	5XA�    �i�A�    �q�A�    <�9qA��    �C�A�l    ��A�t    �T�TA�|    ���A��    �҆A��    ���A�    ��A�    <��,A�    =��NA�    ��A�    �>V!A�    ����A��    ��+�A�    ���kA�    ��w�A�    ��ߙA�    �$��A�    ��iA�    =�X�A�    =f2A��    =��A��    =�Y�A��    ����A��    �{�]A��    =�A��    =�[A��    <�m�A��    =�m�A��    >�`�A��    =��A��    >>z�A��    =���A��    >S��A��    >#�?A��    ���A��    =�%A�    <�e�A��    >{��A��    >3�A��    =��^A��    >T��A��    >ijA��    >@U?A��    =�(kA�    >��cA�    >�9�A�    >	�A�    ��GXA�$    =Y�A��    ���A��    �*Z�A��    ��JA�    �4��A�    ���A�    ��&�A�    �!0A�$    ;��A�,    ��WmA�4    ���A�<    �;�A�D    ��A	     ���LA	     <X��A	     �,rA	 $    <&<�A	 ,    ��xA	 4    =D�6A	 <    ���	A	 D    �(�A	 L    <�l�A	 T    �_�VA	 \    =�}A	 d    ��\A	,    =i>8A	4    =�5+A	<    =]�&A	D    =�B�A	L    =���A	T    =��A	\    =0՘A	d    >.aSA	l    �YA	t    =�`A	|    <M�vA	�    =�F#A	L    ���A	T    =�ogA	\    =��A	d    =0�A	l    =8�A	t    <�W�A	|    =o�A	�    �*��A	�    �q�1A	�    =���A	�    >n"�A	�    >9��A		l    =� �A		t    =��A		|    >�A		�    �*�A		�    >��A		�    >�neA		�    >��A		�    >���A		�    >|{rA		�    >�)UA		�    >Z�#A		�    �GGCA	�    =2�A	�    =��@A	�    >B��A	�    >:�gA	�    >5��A	�    >8cA	�    >��A	�    =��b