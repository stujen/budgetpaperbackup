CDF  �   
      lon       lat       time             CDI       <Climate Data Interface version ?? (http://mpimet.mpg.de/cdi)   Conventions       CF-1.4     history      EThu Nov 03 17:59:45 2016: cdo -a -settaxis,1861-01-16,12:00,1mon -setcalendar,360days -seltimestep,309/2608 tas_Amon_BNU-ESM_piControl_r1i1p1_160001-200812_anom_fldmean.nc CMIP5_IV_4.nc
Tue Apr 12 19:07:18 2016: cdo -fldmean -ymonsub tas_Amon_BNU-ESM_piControl_r1i1p1_160001-200812_short.nc tas_Amon_BNU-ESM_piControl_r1i1p1_160001-200812_ymonmean.nc tas_Amon_BNU-ESM_piControl_r1i1p1_160001-200812_anom_fldmean.nc
Tue Apr 12 12:47:59 2016: cdo -ymonmean -seldate,1600-01-01,2008-12-31 tas_Amon_BNU-ESM_piControl_r1i1p1_145001-200812_all.nc tas_Amon_BNU-ESM_piControl_r1i1p1_160001-200812_ymonmean.nc
Tue Apr 12 12:02:19 2016: cdo mergetime tas_Amon_BNU-ESM_piControl_r1i1p1_145001-200812.nc tas_Amon_BNU-ESM_piControl_r1i1p1_145001-301012_all.nc
2011-11-08T13:31:19Z CMOR rewrote data to comply with CF standards and CMIP5 requirements.      source        BNU-ESM    institution       GCESS,BNU,Beijing,China    institute_id      BNU    experiment_id         	piControl      model_id      BNU-ESM    forcing       Nat    parent_experiment_id      N/A    parent_experiment_rip         r1i1p1     branch_time                  contact       !Ji Duoying (duoyingji@bnu.edu.cn)      initialization_method               physics_version             tracking_id       $525e96fd-c29e-41a4-8c72-60e8d1c0fd7e   product       output     
experiment        pre-industrial control     	frequency         mon    creation_date         2011-11-08T13:31:19Z   
project_id        CMIP5      table_id      >Table Amon (12 November 2010) 6e535ddfacb41fb7a252f4862fdc5766     title         >BNU-ESM model output prepared for CMIP5 pre-industrial control     parent_experiment         N/A    modeling_realm        atmos      realization             cmor_version      2.7.1      CDO       @Climate Data Operators version 1.7.0 (http://mpimet.mpg.de/cdo)          lon                 standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X           |   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y           �   time               standard_name         time   units         month as %Y%m.%f   calendar      360_day    axis      T           �   tas                    	   standard_name         air_temperature    	long_name         Near-Surface Air Temperature   units         K      
_FillValue        `�x�   missing_value         `�x�   comment       1near-surface (usually, 2 meter) air temperature.       cell_methods      time: mean     history       J2011-11-08T13:31:19Z altered by CMOR: Treated scalar dimension: 'height'.      associated_files      �baseURL: http://cmip-pcmdi.llnl.gov/CMIP5/dataLocation gridspecFile: gridspec_atmos_fx_BNU-ESM_piControl_r0i0p0.nc areacella: areacella_fx_BNU-ESM_piControl_r0i0p0.nc          �                A��    �D)�A��    ��U�A��    �k�A��    =��VA��    >*̜A��    �\e1A��    =DkA��    �\�SA��    <�{A��    ;��A��    �L�A�    �f��A��    �b��A��    ����A��    ���VA��    �C͂A��    ���A��    �{UA��    ��B7A�    ��UA�    ��AA�    ��2A�    �;e.A�$    ��ΔA��    ���A��    �\YA��    =�t�A�    >+�pA�    <�%A�    >2�GA�    <��A�$    �b�sA�,    ��f�A�4    �Ɓ?A�<    =>'A�D    <J��A�    ��q�A�    ���A�    ����A�$    ���XA�,    =VގA�4    ��1A�<    ��ߎA�D    �bp\A�L    ��*A�T    ��.�A�\    �P�A�d    �D;�A�,    ���A�4    ��g�A�<    ��A�D    ���A�L    �ž�A�T    ��ԕA�\    ��b�A�d    ���A�l    �DA�t    ����A�|    �+�AĄ    ���0A�L    ��l{A�T    �3͙A�\    =�0A�d    >%MsA�l    ��o�A�t    =�MA�|    =e}�AǄ    =�Aǌ    =���Aǔ    =W۩Aǜ    =��CAǤ    =��|A�l    =�xA�t    �/}A�|    =gk�Aʄ    >~�lAʌ    =�vAʔ    ����Aʜ    ��UAʤ    ��jAʬ    �D�:Aʴ    =���Aʼ    ���+A��    ���LA͌    <�FBA͔    �t͗A͜    =~/Aͤ    =��#Aͬ    >��Aʹ    ����Aͼ    ��A��    �_WA��    <�PA��    ����A��    � ��A��    �;;�AЬ    ���sAд    ��A$Aм    �h�A��    ���kA��    �+�A��    ��y%A��    �M[�A��    ���$A��    �H�DA��    �@�A��    ���A�    ��U�A��    ����A��    �jA��    ��٢A��    �nB�A��    ���A��    =�-@A��    �֍TA�    ����A�    �Q�A�    �9�A�    �+��A�$    ����A��    =��A��    ���A��    =��A�    =��^A�    =�C�A�    �>.A�    �O��A�$    ��+iA�,    �):�A�4    =}��A�<    �H�A�D    ;Ȋ�A�    <�I�A�    ��A�    �`o�A�$    =��A�,    =�~�A�4    =��zA�<    >��A�D    =Sh]A�L    ��ubA�T    ��O�A�\    �aU�A�d    ��{uA�,    �AMA�4    �bY�A�<    �%}�A�D    ��K1A�L    =`�KA�T    �W`�A�\    ����A�d    �v<nA�l    �(?A�t    �o�A�|    �O�A݄    �0B�A�L    �`�A�T    <㴨A�\    �g��A�d    >	3�A�l    >y��A�t    ��2A�|    =��NA��    =��NA��    >(y4A��    =��A��    <��PA�    >E�A�l    >���A�t    >?{XA�|    >-�A�    <�j}A�    >�C�A�    =⿹A�    >Qn�A�    =��A�    >eO�A�    >oխA�    >5;QA��    <��IA�    <�.�A�    �	8A�    �3fnA�    ���RA�    ���$A�    �D��A�    ���A��    ��[�A��    ���A��    �`{aA��    �5˿A��    =��A�    =�0�A�    �i�.A�    �=�A��    <ؓ�A��    =���A��    =�A��    >)��A��    ��w`A��    =�}�A��    =4)�A��    �S�A�    �lA��    ��"%A��    ���\A��    �Fo1A��    >_�aA��    �'�A��    �ǅ:A��    �O�A�    ��O�A�    =ɥ�A�    �p��A�    ;5
�A�$    =jA��    ��#HA��    �%�IA��    <Y�GA�    > ʑA�    ��lA�    ��#A�    ����A�$    ��}A�,    �NO�A�4    ���A�<    ���A�D    ��2�A�    ��6A�    ���"A�    �&p�A�$    ���A�,    �y�@A�4    ��ۅA�<    ����A�D    ��GA�L    �k��A�T    �y0qA�\    ��sA�d    �h(�A�,    =Fq�A�4    ����A�<    ��GA�D    ��B�A�L    >$�A�T    >��A�\    >Lf�A�d    =�?�A�l    �CY�A�t    ���@A�|    >8�A��    �y�A�L    =�	A�T    ���4A�\    <���A�d    ���vA�l    ����A�t    �IA�|    �8dfA��    �D�.A��    ��LA��    =?ֈA��    �\� A��    >a#A�l    =A�t    <,�A�|    =ӌ�A��    =�ğA��    >/��A��    ��<wA��    �%@A��    ���A��    <9��A��    =�TA��    �Ʃ�A��    ��wA��    ;���A��    �4�*A��    �IR�A��    ���kA��    ��c�A��    ���TA��    �e(�A��    �S��A��    �:�A��    ��3�A��    ��A��    ��Z�A�    �~��A�    ��sA�    <���A�    >)j\A�    �tUA�    >��A�    >���A�    ><�A�    > J�A�    �c+ A�    �0j`A    �U�A�    �o�-A�    ��A�    ���A�    ��aoA�    ��A�    �:&jA�    �ÉA    ��A    <�A    ���A    ��R A$    =��KA�    =R!�A�    =4\�A�    =�A	    =�'�A	    >&��A	    >_WLA	    =��KA	$    =���A	,    >n2A	4    �<A	<    �'vA	D    >B߬A    ���A    �k��A    �ٵ�A$    �h��A,    �w�A4    � ��A<    ��PAD    ��CJAL    ���AT    ���NA\    �z�Ad    ����A,    �XA4    ��i�A<    ��ۿAD    =Rh�AL    �z��AT    �c�A\    =�3Ad    =U�Al    �-�}At    ��"�A|    ��`A�    �|	AL    ���=AT    �wA\    ��Ad    ��(Al    ���3At    ����A|    �d�A�    �W`A�    <��tA�    =:k>A�    =�diA�    =�$Al    =�`�At    <̨�A|    =��AA�    �c�5A�    ����A�    >;��A�    >9d1A�    >KM�A�    =�(A�    >v��A�    =��A�    >�&SA�    >2�bA�    �?��A�    =���A�    >TSgA�    >�5A�    =L��A�    =�j�A�    =�t�A�    <�]�A�    =�A�    =���A�    �5A�    ��[uA�    �"kA�    =_FA�    >0n3A�    �t�A�    �Y�A�    ��� A�    �� A�    �ÖA�    ��^-A�    =�A    <�M�A�    �v{A�    ����A�    ��z�A�    �x�A�    ���tA�    >�M�A�    =z��A    >|xA    >+��A    >�{tA    =�#A$    =��{A!�    > S�A!�    =@��A!�    =��A"    ���A"    �p�@A"    �ֆA"    �.InA"$    �=��A",    �"#oA"4    =�ȭA"<    >.�A"D    >R	A%    =Z��A%    �pʿA%    =�GA%$    �i��A%,    ���A%4    ��NeA%<    �c�A%D    ��	�A%L    ���A%T    ���A%\    ��A%d    =�U&A(,    >4!�A(4    >o�9A(<    >2��A(D    =��6A(L    >]XA(T    >B��A(\    <���A(d    =�֥A(l    <�,6A(t    =/&A(|    ����A(�    ���BA+L    =�+MA+T    =o#�A+\    =�u�A+d    �'�.A+l    >pA+t    �lRIA+|    ��A+�    ��A+�    �<��A+�    ���A+�    �}�nA+�    �8B]A.l    �3�A.t    >7 uA.|    �1�uA.�    �g��A.�    �8��A.�    �f�A.�    =�2A.�    ���A.�    ���[A.�    ;���A.�    ��.A.�    ���GA1�    ���A1�    ����A1�    ���A1�    =k�	A1�    >��lA1�    =z�?A1�    >��A1�    >�`�A1�    >M�dA1�    ���IA1�    =2CA1�    >A��A4�    >+��A4�    >q�ZA4�    >A��A4�    >W5A4�    >WU�A4�    >1:�A4�    >���A4�    ���A4�    >XQA4�    =/�jA4�    ��u�A5    �(s	A7�    � �A7�    ���5A7�    >6�A7�    �і5A7�    �Ԥ!A7�    ��S A7�    ��*A8    ��A8    ���{A8    ��D8A8    �09A8$    �Ū�A:�    ��A:�    ��H�A:�    �q}�A;    ����A;    ���uA;    ���A;    ����A;$    ��W�A;,    ��f}A;4    �SmA;<    �D�0A;D    ���A>    ��A>    �L\xA>    �$$�A>$    ���QA>,    ��*ZA>4    ����A><    =(�A>D    <,"�A>L    =�CA>T    <��A>\    ��A>d    �{�AA,    ���AA4    >$�AA<    =">nAAD    >L��AAL    ���AAT    �oϖAA\    =AAAd    >]�QAAl    �[�AAt    <ܕTAA|    ���AA�    <���ADL    =�[�ADT    ��0AD\    >1ADd    �B��ADl    �8eADt    �pFAD|    ���AD�    <��AD�    ��AD�    ��uAD�    ���AD�    ��AGl    ��& AGt    �D)�AG|    ���RAG�    ��fVAG�    ����AG�    �*4nAG�    � �%AG�    ����AG�    �N0�AG�    �&��AG�    ��)#AG�    ���AJ�    ��nAJ�    =�!�AJ�    ="_�AJ�    =�V�AJ�    =��wAJ�    �P�AJ�    �J�AJ�    ����AJ�    <�$AJ�    ��֙AJ�    �N�AJ�    =��AM�    ��6�AM�    <J-fAM�    ��@AM�    �@$TAM�    ��z�AM�    ���\AM�    =�IAM�    >Z�AM�    >4]AM�    =�i�AM�    =Bu�AN    <�2AP�    =H�AP�    =� AP�    =��AP�    >w��AP�    >{AP�    >�f9AP�    >�-AQ    ��R�AQ    �BAQ    =�_wAQ    ��"(AQ$    =�q,AS�    >&x�AS�    >��AS�    >���AT    ?��AT    ?&�AT    ?%<�AT    >���AT$    >�WSAT,    >�|AT4    >��\AT<    >�}�ATD    =�luAW    >^��AW    ��TAW    >2@eAW$    >�;AW,    �Ғ�AW4    =�,�AW<    =_�AWD    ���
AWL    ��AWT    �k�AW\    ����AWd    �ZBAZ,    <�p�AZ4    ���!AZ<    :R��AZD    �RAZL    >@�AZT    =�H$AZ\    =���AZd    =�zuAZl    =��SAZt    =��@AZ|    <Q�AZ�    =���A]L    �VV�A]T    =�Z�A]\    �&A]d    �s0.A]l    =�O�A]t    =��A]|    <B�A]�    �-j�A]�    ���A]�    =���A]�    >,��A]�    >�A`l    =���A`t    ���A`|    �2�.A`�    ����A`�    ��v�A`�    ����A`�    �IבA`�    �?�:A`�    �L��A`�    �J*�A`�    ��A`�    �62+Ac�    �6��Ac�    �0�,Ac�    ��ctAc�    ��m�Ac�    ��.lAc�    ��iAc�    ��e�Ac�    ���Ac�    ��Ac�    �	��Ac�    �P��Ac�    ���Af�    �NY�Af�    �e�vAf�    ���Af�    �*>Af�    �21Af�    ����Af�    ���Af�    �/�SAf�    �5}�Af�    �UŞAf�    �+�CAg    �X�Ai�    ��XAi�    ���#Ai�    �L[�Ai�    ��obAi�    <�y�Ai�    �=7�Ai�    �P}\Aj    ��̙Aj    �Y1UAj    � �Aj    ��QAj$    �g�>Al�    ��\Al�    ����Al�    � ��Am    �ʥ�Am    �>�Am    �)ăAm    �elZAm$    ��/AAm,    ��	Am4    �n\dAm<    ��+�AmD    ���[Ap    ���Ap    �k��Ap    ����Ap$    �H�HAp,    =�^�Ap4    �"�nAp<    =�!�ApD    �Y�HApL    ���5ApT    ���Ap\    �G�DApd    ��X�As,    �'�As4    >`I�As<    ��͏AsD    �j/lAsL    �ѓyAsT    �]��As\    ���Asd    ����Asl    �-SAst    ���nAs|    <�zAs�    >WˬAvL    =��"AvT    =:	�Av\    >��Avd    =�5�Avl    >5޾Avt    >��Av|    >G�OAv�    >��Av�    >?�(Av�    >��|Av�    >��Av�    =r��Ayl    =[��Ayt    >n0�Ay|    ��U�Ay�    �H��Ay�    >@�Ay�    >���Ay�    >��Ay�    >-QAy�    =��!Ay�    <�}+Ay�    ��s�Ay�    = �A|�    �ЬA|�    �
)+A|�    ����A|�    ���A|�    ��+A|�    ��fuA|�    ���jA|�    �4x�A|�    ����A|�    �m�dA|�    ���A|�    �Ǔ�A�    ��Z�A�    �#>`A�    ��90A�    <k��A�    <��(A�    =��A�    >��A�    >$�\A�    =�u%A�    =ʈ�A�    =C8tA�    ;s�.A��    =���A��    ��A��    �A{A��    ��<A��    �ϲ8A��    ��CEA��    ��R�A�    ��W=A�    �3�A�    �eA�    �"H/A�$    ���A��    ����A��    �\hgA��    ���{A�    >��A�    ���A�    >���A�    >~%8A�$    >	A�,    >*A�4    <�S�A�<    <FyoA�D    >,N#A�    >p	A�    >�*vA�    ����A�$    <��tA�,    >�[A�4    >��A�<    <�{LA�D    >3aFA�L    >Z�A�T    >GjYA�\    ;�S�A�d    >	 �A�,    =��A�4    ����A�<    ���A�D    =��HA�L    �=�A�T    ��_A�\    �EީA�d    ��P�A�l    ����A�t    ��wA�|    �F�4A��    ;�ןA�L    <��A�T    ��dPA�\    <��A�d    ��
�A�l    =�L�A�t    =P�A�|    �`��A��    =8�A��    �z��A��    ��#�A��    =\�A��    =�[A�l    =��A�t    >_)�A�|    >���A��    > �A��    =�2A��    >�:A��    >�8AA��    >�doA��    >��SA��    >�#A��    =���A��    > �A��    <�7�A��    ���A��    ���}A��    �%%A��    ����A��    �jCA��    � #A��    �ĠA��    ���A��    ��!A��    �`^A��    ��A��    �G�A��    =_�A��    ���A��    ���-A��    ���iA��    ��W�A��    =�RA��    =��A��    ��3A��    �)�A��    ;�NA�    =�aA��    >\k�A��    �isA��    >�I�A��    >�ǸA��    >*��A��    >���A��    >�:zA�    ?��A�    >��VA�    >��A�    >���A�$    >���A��    >7��A��    =�:�A��    =cHvA�    ��!�A�    ��"�A�    >���A�    =��9A�$    <�&�A�,    >P�A�4    �kI�A�<    ����A�D    ��rA�    �z@A�    �'wtA�    ��m3A�$    ���PA�,    �>uA�4    ���A�<    ���iA�D    =��{A�L    ={�jA�T    �	�"A�\    ��|A�d    =0��A�,    =���A�4    =i�A�<    ��oA�D    >Ox�A�L    =5�
A�T    �[��A�\    <x7�A�d    <ѽQA�l    =��A�t    =���A�|    =+�&A��    ���QA�L    ��gA�T    ���:A�\    ��]�A�d    ���
A�l    >+��A�t    >�>&A�|    <��A��    >-��A��    �-�A��    =���A��    =�tnA��    >�A�l    =`q$A�t    =�MA�|    >0`�A��    >s�A��    =VA��    >"�5A��    ��:A��    <��A��    �� xA��    ����A��    >�A��    �U�A��    �l9A��    � KA��    ��OA��    =sN�A��    >h@�A��    �7{8A��    �b�A��    ��VA��    �a��A��    �I�GA��    ���UA��    =���A��    ���A��    =Hh�A��    �\[�A��    > n�A��    >bkA��    =�_A��    �#]A��    �^��A��    �wSA��    ��!�A��    ��LA�    <�~A��    ����A��    �w�A��    =�A��    ���A��    =�\A��    =�gA��    �\��A�    =ٚ�A�    ��/�A�    =��A�    ��/A�$    ����A��    ��h�A��    ��A��    =uHA�    ��stA�    �G�A�    =x�vA�    <.5"A�$    ����A�,    �nf4A�4    ���EA�<    ���eA�D    �k��A�    �ZEA�    �DF.A�    ��A�$    ��s�A�,    ����A�4    <�2`A�<    �1RZA�D    =^��A�L    =���A�T    ��QA�\    = zCA�d    �+5A�,    >wH�A�4    >KS]A�<    ��$�A�D    =(�A�L    =���A�T    �A	�A�\    =�sA�d    >-��A�l    ���RA�t    �ЦA�|    �=��A��    �C�A�L    ��Q�A�T    �p��A�\    =���A�d    �o��A�l    ��+A�t    =�0A�|    =��[A��    =ӂcA��    �\�A��    >l�^A��    >��A��    =B�;A�l    =�LA�t    >��A�|    >G�AĄ    ���AČ    =�ZdAĔ    =fj�AĜ    <15AĤ    �"p�AĬ    ��O�AĴ    >@87Aļ    >$hA��    =;�Aǌ    �"JAǔ    =�>�Aǜ    >y�AǤ    =�{tAǬ    >GJ�AǴ    =�NpAǼ    �dXA��    <UA��    ���A��    >�jA��    ��W�A��    ��0iAʬ    �q�Aʴ    =J�Aʼ    >:�/A��    >��eA��    =��iA��    >�A��    =�$A��    =�8A��    �F?�A��    �s��A��    ����A�    ��ϐA��    �s_LA��    =���A��    >��.A��    >,wA��    =�5NA��    =! �A��    >A�    ;�OA�    ���oA�    =�;�A�    =��+A�$    �GlkA��    ��� A��    ��c�A��    <��~A�    ���mA�    ��A�    �[�A�    =��A�$    ��`A�,    ��yA�4    ��7
A�<    ��'�A�D    �c��A�    �C��A�    ����A�    ���A�$    �O��A�,    �ϛ�A�4    �p��A�<    ��$A�D    ����A�L    ����A�T    ���FA�\    ��~PA�d    =���A�,    =عvA�4    <�wA�<    =��A�D    =��A�L    >iM�A�T    >LdA�\    >l�A�d    >&��A�l    >d�A�t    >>fEA�|    ��tAׄ    �pȣA�L    =�7�A�T    ��;�A�\    ;�,�A�d    ��y=A�l    �g��A�t    �QHdA�|    �(�<Aڄ    ���Aڌ    ��\Aڔ    ��6HAڜ    ��A�Aڤ    ��=�A�l    ����A�t    ��@A�|    =���A݄    ��A݌    <�MsAݔ    �#�Aݜ    >OxAݤ    ;6QAݬ    �J��Aݴ    =Q��Aݼ    ���A��    �);uA��    >��A��    ���^A��    >*z�A�    <���A�    =��A�    ��+A�    ��uA��    ���EA��    �4��A��    ���JA��    ����A��    >zL=A�    >�	A�    =��
A�    >#�A��    =�>�A��    >V��A��    >�2]A��    =�gkA��    =�LA��    =���A��    <��PA��    =�&�A�    ��D1A��    ��A��    :)	BA��    ��ƹA��    �1��A��    <�QA��    �Ý<A��    <�<A�    >X]A�    ��1A�    ����A�    � ��A�$    �D'�A��    �R=3A��    ��A��    �B�SA�    �D�A�    �MOkA�    �	�A�    �w[A�$    ���<A�,    ��T)A�4    �8��A�<    �� A�D    ��w�A�    =��5A�    =~��A�    ���A�$    �Z߲A�,    =֒OA�4    =c�jA�<    �
�%A�D    :�KzA�L    <}��A�T    ��tA�\    =\�:A�d    ����A�,    ����A�4    �NZ�A�<    ���A�D    �IA�L    =V	A�T    =7�A�\    =��A�d    ��G�A�l    =��A�t    <�v�A�|    <�R�A��    =��A�L    ��dA�T    =���A�\    =`cA�d    �5�A�l    =V8�A�t    ��v�A�|    <L��A�    �U$A�    � �A�    �,�=A�    ����A�    �FڑA�l    =o=A�t    =���A�|    ��)PA��    ��jA��    ��A��    ���VA��    �rM~A��    ���A��    �J��A��    �1��A��    ��;A��    �	��A��    �:��A��    �k�%A��    ����A��    �q�xA��    �(��A��    ���A��    �ڞ�A��    �&�lA��    �O��A��    �,l�A��    ��y�A��    ��;FA��    ����A��    =kQpA��    ����A��    ���A��    >���A��    =,;A��    ��FA��    �CnVA��    �L�A��    ����A��    ��7�A�    �ph�A��    �J�A��    ��r�A��    ����A��    ��?A��    >�TA��    �B��A��    ��RA     �c۪A     <��A     >8�A     >*�A $    >cXA�    >_�|A�    >��A�    �^�A    =� �A    �h��A    ���A    >eRA$    =�VA,    =@4xA4    >k�A<    ��$3AD    =�vA    < �A    >~A    �３A$    =ז�A,    =u��A4    � n�A<    ����AD    �#�AL    �5J'AT    �پ[A\    �u.JAd    �"ȆA	,    �"y�A	4    �E�A	<    ���A	D    =0P�A	L    ���[A	T    �E��A	\    �&�6A	d    ��c�A	l    ���!A	t    <��eA	|    ��ZUA	�    ���AL    ���AT    �a�?A\    �X϶Ad    =�*�Al    =�BAt    =���A|    �FybA�    �ɩ�A�    ��ȈA�    �̧lA�    =P�xA�    <C�(Al    =�ٴAt    <˥A|    =g��A�    =��
A�    >,M�A�    >>P�A�    >��SA�    >W��A�    >/�.A�    >��RA�    >���A�    >=ucA�    >��A�    >��fA�    >K(�A�    =�t]A�    >OBA�    >F�A�    =�MA�    <��A�    ���ZA�    ��]�A�    �T,A�    �8TPA�    ��k>A�    �2`^A�    �
l=A�    �E��A�    �b��A�    ��E�A�    �Z�^A�    �2��A�    =.�A�    �X��A�    ��,�A    =�_A�    >��A�    >G�9A�    >��cA�    =��A�    ��$A�    >�Y(A�    >d��A    >�frA    >C�A    =V\�A    <�vA$    ;�gA�    �z<|A�    =�Q A�    ���A    �@|A    <?�]A    =�(GA    ����A$    �>5�A,    �ȭ�A4    �cjA<    �`� AD    <��A    =���A    =q�A    ����A$    =�bA,    >�F�A4    >���A<    >�@AD    >iOAL    ;>˧AT    ���eA\    �6g_Ad    ��ʡA",    ����A"4    �w��A"<    ����A"D    =ZN�A"L    �0A"T    =�]�A"\    ����A"d    ��?�A"l    ��R�A"t    �ƣA"|    �E�A"�    =���A%L    =�<!A%T    >*��A%\    =�υA%d    >���A%l    :EA%t    >���A%|    >	��A%�    >�aA%�    <��NA%�    ����A%�    <��A%�    ��DA(l    �hБA(t    =QN	A(|    =4�A(�    ���A(�    �-�A(�    >��A(�    =� uA(�    ��D�A(�    =��qA(�    =<��A(�    =/�A(�    =�yA+�    ==#PA+�    �ƯA+�    �Q.A+�    >.j2A+�    ����A+�    ��GnA+�    ;��A+�    =��A+�    >���A+�    >��A+�    >��+A+�    >r׻A.�    >hY�A.�    >f�cA.�    >���A.�    >u?QA.�    >�N�A.�    >R WA.�    >_A.�    �Н�A.�    ���JA.�    =�V�A.�    ��\�A/    �M�?A1�    �'��A1�    �Z�AA1�    �S+�A1�    ���MA1�    ��#6A1�    �jXrA1�    �ǙA2    ���=A2    ��zxA2    ��"�A2    ���&A2$    ��O�A4�    �,7�A4�    ����A4�    �|x�A5    ��&A5    �j_~A5    ���A5    ����A5$    ��МA5,    �ޢA54    ���A5<    �Fo�A5D    ���A8    �|K�A8    ���A8    �G�A8$    ����A8,    ��Q A84    �>�A8<    ���DA8D    �@��A8L    �[�A8T    ��(#A8\    ���A8d    ��<A;,    �'��A;4    ��A;<    >5A;D    ��_�A;L    =�D�A;T    >dA;\    >3�_A;d    �I(OA;l    ���A;t    >/o�A;|    =�A;�    <���A>L    =�A>T    <�tiA>\    =i�A>d    >m��A>l    >S�A>t    =��/A>|    =��A>�    ��)A>�    �4a�A>�    �%=�A>�    ���A>�    �='AAl    ��s�AAt    �;JAA|    ���4AA�    ��o�AA�    ����AA�    �FT�AA�    ��&�AA�    ��!�AA�    �p�sAA�    �C*AA�    �R@�AA�    ��GAD�    ��AD�    ��7AD�    <���AD�    �{� AD�    =���AD�    >�o/AD�    �2��AD�    �1�QAD�    �Q`AD�    ��xjAD�    ���AD�    �P9�AG�    ��4AG�    ��[�AG�    ��PRAG�    ��x�AG�    <���AG�    ��lAG�    ���+AG�    ����AG�    �^FhAG�    �}��AG�    ��t�AH    =�g�AJ�    =�g'AJ�    ���AJ�    ���}AJ�    >��AJ�    >1�zAJ�    >T]AJ�    =��FAK    >`'AK    >L	AK    =��AK    �@cYAK$    �<�AM�    <���AM�    �=�OAM�    �:�fAN    ���EAN    �4�AN    ��BAN    ��s�AN$    ���#AN,    ���AN4    �D��AN<    �93�AND    ���AQ    �^��AQ    �ƁPAQ    �2{AQ$    =���AQ,    =���AQ4    >�C3AQ<    >C�'AQD    >��6AQL    >:�AQT    �q[�AQ\    �/��AQd    =���AT,    < |AT4    <ԗWAT<    �&<�ATD    <pEsATL    >r�4ATT    =��AT\    >Y��ATd    ��ATl    �4�lATt    �l�8AT|    ���2AT�    �C�YAWL    �B&KAWT    ���AW\    ��T�AWd    ��h�AWl    ��aAWt    �y��AW|    ���QAW�    ���*AW�    �ȍiAW�    �u+?AW�    ��HAW�    ��ĉAZl    �۞�AZt    �CăAZ|    =]1�AZ�    ���RAZ�    <Ǹ�AZ�    <��}AZ�    ����AZ�    =��pAZ�    ���AZ�    �>O�AZ�    �{��AZ�    �giBA]�    �2v�A]�    �?U�A]�    �G�A]�    =��AA]�    �7�dA]�    ����A]�    ��WEA]�    ���A]�    ���;A]�    �=�3A]�    �KuVA]�    �- �A`�    �ɌA`�    ����A`�    �o��A`�    �"A`�    =�,�A`�    �ީ�A`�    �9��A`�    ���7A`�    ;�/�A`�    >OO�A`�    <�+Aa    ����Ac�    =@7Ac�    =]�*Ac�    >�@�Ac�    <�{Ac�    >J��Ac�    =�rAc�    >yMAd    >
� Ad    ;��Ad    >4,Ad    =���Ad$    >�?Af�    =ɻ)Af�    >�Af�    >yF+Ag    >QnAg    >�I^Ag    >� �Ag    >�A�Ag$    >��pAg,    >��eAg4    >���Ag<    >��AgD    >�5�Aj    >�L�Aj    >��Aj    >��Aj$    >�>Aj,    >�:Aj4    ?7�9Aj<    ?��AjD    ?=AjL    >���AjT    >t��Aj\    >o
�Ajd    >~WAm,    =���Am4    ���TAm<    ��o�AmD    �><AmL    ��K�AmT    <��Am\    ;�L�Amd    �ޓLAml    �)LAmt    ����Am|    ��tNAm�    ��>ApL    =��ApT    ��Ap\    ��<Apd    =:�Apl    >�U�Apt    >s��Ap|    ���CAp�    >�ayAp�    =�0GAp�    >gjAp�    >HO�Ap�    ��~vAsl    ���Ast    �.�;As|    ����As�    ���As�    �T�4As�    �|$As�    �g�(As�    ��As�    ��uYAs�    ���As�    ���As�    ��Av�    ��/Av�    >#b�Av�    >Ft`Av�    =_�Av�    >;�:Av�    =��Av�    >~��Av�    >�6vAv�    >@,Av�    >�h�Av�    >�v�Av�    =�7\Ay�    > �PAy�    =��Ay�    <�Ay�    ����Ay�    �2a3Ay�    �#�Ay�    ��I-Ay�    �ز(Ay�    �45 Ay�    �t��Ay�    ���Az    �A��A|�    ��'�A|�    ���"A|�    <�,�A|�    =AA|�    >	ήA|�    �'�A|�    �=ԦA}    =�v`A}    >u�A}    =~^A}    ��<sA}$    <i
�A�    <�4�A�    =e�4A�    =�@�A�    >/�7A�    �̕KA�    >@�A�    =�]�A�$    =�eA�,    >�A�4    =:��A�<    >a��A�D    >T=A�    �y�A�    =���A�    =�"A�$    :�	A�,    ���A�4    �%�A�<    �!9A�D    ��ECA�L    �p߮A�T    �0A�\    ��%A�d    ����A�,    ���A�4    ����A�<    ���A�D    �ڏlA�L    =�g�A�T    �Z`�A�\    ��ڤA�d    ���A�l    �N�vA�t    �9�4A�|    ���$A��    > �FA�L    =�@A�T    >`h@A�\    >!I�A�d    ���A�l    >6��A�t    >�]A�|    =ߜ�A��    >��A��    =��JA��    >R�3A��    >�A��    =�YA�l    <]V�A�t    ��yA�|    > A��    =���A��    <�=A��    =,A��    =j��A��    ��c�A��    �>�NA��    �;��A��    �(j�A��    �5WA��    �MHA��    �4%�A��    ��^�A��    ��|�A��    ����A��    ����A��    �OoA��    ��;6A��    �7'�A��    ����A��    >Wt	A��    =q�A��    =x�MA��    =�/�A��    � ��A��    >!�4A��    ���A��    =0LA��    �[��A��    ���qA��    ��g�A��    �A��A��    �/�A�    ��g3A��    �.L�A��    ��~�A��    �?�0A��    �M�A��    ���~A��    �5-�A��    �"AA�    ���A�    ��( A�    ��ŭA�    �)�A�$    =rkA��    ��TWA��    ><��A��    ��m�A�    >�)A�    ����A�    ���A�    <n-A�$    >:dNA�,    �bxA�4    ����A�<    =#i�A�D    ���A�    =p�A�    =�jJA�    =,��A�$    ����A�,    ��W�A�4    ��PA�<    ���fA�D    ��d^A�L    ���TA�T    �(:A�\    ���A�d    ���kA�,    �-��A�4    =�G�A�<    >	�]A�D    �c0�A�L    ���A�T    �)e�A�\    >C.A�d    ��A�l    ��V�A�t    =�]A�|    >9�A��    ���A�L    �#�A�T    =�W*A�\    =;�6A�d    ��l�A�l    >l A�t    =�ǳA�|    =���A��    =���A��    <:�A��    ��X�A��    ���A��    �u+A�l    ����A�t    �Y*�A�|    ��o�A��    ��k0A��    �PA��    ����A��    ��ҘA��    ����A��    ����A��    ��aUA��    �H
�A��    �H�'A��    ��wA��    >�EA��    =��A��    �F��A��    =��A��    ���A��    �.J�A��    =�_aA��    >9K�A��    >�gA��    >���A��    >3||A��    >n�A��    >a�A��    =�"A��    =��`A��    ����A��    �;�A��    �7A��    >$Y�A��    =1��A��    =�A��    �d�A�    =&��A��    ��<[A��    ��A��    ���rA��    ��\�A��    �h�^A��    ���A��    ��`^A�    ��AA�    ��XA�    �|P�A�    �>��A�$    �^��A��    �XbA��    ����A��    �U�oA�    ��\�A�    >l��A�    ��F�A�    <��A�$    ��A�,    ��A�4    =�"0A�<    ����A�D    =�1�A�    =��A�    >�,�A�    >���A�$    >�KvA�,    ���NA�4    >��gA�<    >�VpA�D    =d"A�L    �}_A�T    =��A�\    �*lA�d    ��/A�,    ����A�4    � �ZA�<    ����A�D    �^C�A�L    �p�A�T    ��aA�\    ����A�d    �O}�A�l    ��b�A�t    �K^?A�|    �p�fA��    ����A�L    ��lA�T    �"׿A�\    ��B�A�d    ����A�l    ����A�t    �NX�A�|    <��LA��    =��A��    =��zA��    �<$�A��    =@0A��    >�HfA�l    >�}�A�t    >o�A�|    >"�A��    >/�xA��    > �A��    ?�SA��    ?��A��    >ǟ�A��    =��XA��    >�A��    >�D�A��    >qK�A��    >1�xA��    =��7A��    >?�A��    �4�QA��    �p�?A��    =+, A��    =��A��    �eD�A��    ;��A��    <�JA��    ��A��    ����AĬ    <�vFAĴ    �0�Aļ    =��]A��    ��dA��    >DA��    ��o�A��    ��C�A��    �0S�A��    � �A��    ��kA��    =�.A�    =ZA��    �_��A��    ���A��    =�9A��    ���A��    ���A��    ��A��    �͌�A�    =���A�    <��9A�    �̒YA�    =}�A�$    >+�A��    <2ZA��    <�(A��    ��}A�    ���A�    ��7�A�    �M)�A�    >:��A�$    =�XA�,    =�څA�4    =��QA�<    >+�&A�D    >!��A�    =�ʹA�    =�K�A�    �(��A�$    =�9IA�,    =���A�4    =��A�<    =7h�A�D    =��YA�L    =0ujA�T    �iʄA�\    ���zA�d    �o�A�,    �X#�A�4    ���.A�<    �Z=A�D    �r��A�L    ���dA�T    ���jA�\    �W�BA�d    ����A�l    �)�A�t    �.�A�|    �kb$Aф    �L�7A�L    ����A�T    ��A�\    ���A�d    �3�hA�l    ���A�t    ;�!�A�|    ���AԄ    �LAԌ    �!��AԔ    ��G:AԜ    =� �AԤ    �(JA�l    �dd�A�t    ��A�|    =\i�Aׄ    ���A׌    ��%�Aה    ���+Aל    ����Aפ    �?rA׬    �_dtA״    �Ao�A׼    =��A��    ���vAڌ    � ��Aڔ    �E}�Aڜ    ����Aڤ    >�<Aڬ    �i'-Aڴ    ���Aڼ    �F�A��    <�ݦA��    �I��A��    ����A��    ��VNA��    <?,�Aݬ    ���Aݴ    �{f�Aݼ    �&-tA��    �
��A��    >rA��    �^�A��    �:��A��    �,{�A��    �:A��    �I}A��    �w	GA�    �bA��    ���sA��    =��xA��    >KvzA��    =�A��    �ٌ�A��    �V�vA��    �d�A�    ��,A�    =���A�    >�/_A�    =���A�$    >s�A��    >��A��    >n8�A��    >���A�    >��8A�    >���A�    >�NA�    >kb�A�$    >O��A�,    >�ĐA�4    >f�|A�<    =���A�D    �G_�A�    �78�A�    �O�.A�    �RaA�$    ��D�A�,    ���A�4    ���YA�<    �p��A�D    ����A�L    �+�A�T    ��VA�\    ���cA�d    ��&�A�,    �� �A�4    �7�dA�<    ���&A�D    ���A�L    �i�`A�T    ��uA�\    ��FlA�d    ��A�l    ��l>A�t    �q�7A�|    �cA�    �*w�A�L    ��4�A�T    ��wA�\    ���A�d    ���GA�l    �'�hA�t    <c�A�|    ���
A�    �
�cA�    <EGA�    =��A�    =�=,A��    >C��A�l    �8lA�t    ����A�|    �ޯIA��    =Ϭ�A��    ����A�    �6�A�    ��9hA�    ��'�A�    =���A�    >a/A�    >��RA��    >��A�    �`�A�    ��	A�    ����A�    ;��EA�    >�+MA�    ����A�    ��r�A��    <J�A��    =�rA��    =""bA��    =��A��    >7��A��    =حvA��    >b�A��    >[p0A��    =��XA��    >\A��    =��A��    >C�A��    >KZ�A��    >A�^A��    �PA��    �nNA�    =�MA��    ���KA��    ���A��    �z��A��    ���A��    �{
�A��    ��]A��    ��اA�    ���A�    �LLA�    �A�IA�    ����A�$    =P*�A��    ��8cA��    ��&KA��    �04�A�    <��A�    ��	�A�    �t��A�    =7A�$    >\��A�,    >E0KA�4    >�Y�A�<    >?LA�D    >�L�A	     =�m�A	     >��A	     >�d2A	 $    >�PwA	 ,    >c��A	 4    >rA	 <    >TD�A	 D    >4z�A	 L    >!��A	 T    >���A	 \    ��m�A	 d    �ݳ3A	,    �>��A	4    >��A	<    <��rA	D    ���A	L    ����A	T    �~��A	\    ��dA	d    ����A	l    ��A	t    �uA	|    �rbyA	�    ��-A	L    �=�hA	T    ��]wA	\    ���A	d    ��:�A	l    ��'A	t    ��M�A	|    ��&HA	�    �|��A	�    ���A	�    ���A	�    ����A	�    �s:�A		l    �vsA		t    �$�4A		|    >4AA		�    =گ�A		�    �qb�A		�    ��A		�    ��}fA		�    >W�A		�    ���TA		�    < �A		�    >�A		�    �Y�(A	�    =몛A	�    �äA	�    ����A	�    �ɿ�A	�    =��JA	�    =��A	�    >_��A	�    =ѥ