from __future__ import print_function
"""
DataPlan Class
read either a python or excel ('xlsx') file, and return a dict with the "dataplan"
Note that some specific requirements are present in the current version regarding
the names of the keys and the columns.

A dataplan must have:
datasets (dict)
basepath : the base path to the data files
outputpath : the path to where the output files of an analysis would be stored.

In the excel file, there should be one sheet "Directories" with a directory column
and antoher sheet "Dataplan" with at least Date, Slice, Cell, Protocols and a few other entries,

"""

import os
import numpy as np
import pandas as pd
#import ezodf as odf
from collections import OrderedDict
import re


class DataPlan():
    def __init__(self, datadictname, sheet='Dataplan'):
        """
        set up and read a data plan, saving the structure and information for other uses
        
        Parameters
        ----------
        datadictname : str (no default)
            name of the file that holds the data dictionary
            Should end in .py or .xlsx. The file is parsed to a common structure
        
        sheet : str (default "Dataplan")
            The name of the sheet in the excel file that will be used (if an excel file is specified)
        
        Returns
        -------
            Nothing. All results are in class variables
        """
        
        self.sheet = sheet # for excel files
        self.orderedkeys = ['subject', 'dir', 'G', 'prots', 'thr', 'rt', 'decay', 'exclist']  # default set
        data = {}
        fn, ext = os.path.splitext(datadictname)
        if ext == '':
            ext = '.py'
        if ext == '.py':
            h = open(fn+ext).read()
            print('h: ', h)
            exec(open(fn+ext).read(), data)
            #execfile(fn + ext, data)  old python 2.7 version
            self.datasource = datadictname
            self.datasets = data['datasets']  # convenience
            self.datadir = data['basepath']
            self.outputpath = data['outputpath']
            self.data = data  # just save the dict for anything else
        elif ext == '.xlsx':
            self.datasets = self.read_xlsx_datasummary(fn + ext, sheet=sheet)
    
    def setkeys(self, keylist):
        """
        Change the ordered key list values (column names in excel sheet)
        """
        
        self.orderedkeys = keylist

    def make_xls(self, dataset, outfile='test.xlsx', sheet='Dataplan'):
        """
        Process .py files, convert to an excel sheet
        From the dictionary in dataset, write an excel spreadsheet as follows:
        Top keys are rows
        Within each row, keys are columns
        prots and exclist are written as strings of python code
        All others are written as values
        The column headings are derived from the first entry in the data set.
        If a value is missing for a given row, it is left empty
        if a new key is encountered, a new column is made
        
        Parameters
        ----------
        dataset : dict
            Dictionary specifying the dataset (for .py file)
        
        outfile : str (default "test.xlsx")
            Output excel file to create from this dataset
        
        sheet : str (default "Dataplan")
            Sheet in the excel file where the plan will be written
        
        Retuns
        ------
        Nothing
        
        """

        subjectkeys = dataset.keys()
        colkeys = dataset[subjectkeys[0]].keys()
        # append unordered keys now
        for key in colkeys:
            if key not in self.orderedkeys:
                self.orderedkeys.append(key)

        # for each column determine maximum width of field and set it
        cwidths = OrderedDict()
        for col in self.orderedkeys:
            cwidths[col] = 0
            for row in dataset.keys():
                if col == 'subject':
                    dsl = 12
                else:
                    dsl = len(str(dataset[row][col]))
                if dsl > cwidths[col]:
                    cwidths[col] = dsl
        orderedkeys = self.orderedkeys[1:]    
        df = pd.DataFrame(dataset).transpose()
        df = df[self.orderedkeys]
        
        writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
        df.to_excel(writer, sheet_name=sheet)
        wks = writer.sheets['Dataplan']
        for i, c in enumerate(self.orderedkeys):
            wks.set_column(i+1, i+1, cwidths[c]+2)
        wks.set_column(0, 0, 12)
        writer.save()
        self.read_sheet(self.outfile, sheet)

    def read_sheet(self, filename, sheet=0):
        """
        Read an excel sheet and return as a pandas data structure
        
        Parameters
        ----------
        filename : str
            name of the excel file to read (including extension)
        sheet : int or str (defalt: 0)
            sheet within the excel file to read
        
        Returns
        --------
        Nothing
        """
        
        d = pd.read_excel(filename, sheet_name=sheet).transpose()
        ds = d.to_dict()
        for s in ds.keys():
            ds[s]['prots'] = eval(ds[s]['prots'])
            ds[s]['exclist'] = eval(ds[s]['exclist'])
        return(ds)

    def read_xlsx_datasummary(self, filename, sheet):
        """
        Read a datasummary file and store into an pandas dataframe
        Datasummaries are flat text files in a csv format and are generated
        by datasummary.py
        
        Parameters
        ----------
        filename : str (no default)
            Name of the text file to read, including path and extension
        
        Returns
        -------
        dataset as a pandas datafram
        """
        
        re_plist = re.compile(r'([\w\(\)]+)+')
        re_sst = re.compile(r'Protocols: \[([\w\(\), ])*')
        self.excel_as_df = pd.read_excel(filename, sheet_name=sheet)
        ds = self.excel_as_df.transpose().to_dict()
        for s in ds.keys():
            ds[s]['exclist'] = ('{0:s}.{1:s}.{2:s}'.format(str(ds[s]['Date'])[:-1], str(ds[s]['Slice']), str(ds[s]['Cell'])))
            try:
                ds[s]['prots'] = eval(ds[s]['Protocols']).strip()
                ds[s]['IV'] = eval(ds[s]['IV']).strip()
                ds[s]['Map'] = eval(ds[s]['Map']).strip()
            except:
                matchstring = re_sst.match(str(ds[s]['Protocols']))
                if matchstring:
                    res = re.findall(re_plist, matchstring.group(0))
                ds[s]['prots'] = res[1:]
#                print(ds[s].keys())
                ds[s]['IV'] = str(ds[s]['IV']).strip()
                ds[s]['Map'] = str(ds[s]['Map']).strip()
                #print('protos:')
        return(ds)

    def add_result_columns(self, colnames, datatype='float'):
        """
        Add columns to the pandas data frame if they do not already exist
        
        Parameters
        ----------
        colname : str (no default)
            the text namd for the column to add at the right hand end of the sheet
        datatype : str (default: 'float')
            data type for the column (must be float or str)
        
        Returns
        -------
        Nothing. The current pandas dataframe is modified
        """
        
        dflength = len(self.excel_as_df['CellID'])
        for colname in colnames:
            if not self.excel_as_df.columns.isin([colname]).any():
                if datatype == 'float':
                    self.excel_as_df[colname] = pd.Series(np.zeros(dflength), index=self.excel_as_df.index)
                elif datatype == 'str':
                    self.excel_as_df[colname] = pd.Series(['' for x in range(dflength)], index=self.excel_as_df.index)
                else:
                    raise ValueError('add result column needs datatype of "float" or "str", got: %s' % str(datatypepyt))

    def post_result(self, dataname, dataid, colname, value):
        """
        Put results into a cell in the pandas dataframe. If the column
        does not already exist, it is added to the dataframe.
        
        Parameters
        ----------
        dataname : str
            Idendtifier for the data namoing scheme (usuall 'cell')
        dataid : int
            Number/data identification associated with the dataname
        colname : str
            Name of the column to store the data
        value : float or str
            data to be stored.
        
        Returns
        -------
        Nothing. The pandas dataframe is modified
        """
        
        if dataid not in self.excel_as_df[dataname].values:
            raise ValueError('%s number %d is not found in current frame/excel sheet' % (dataname, dataid))
        if not self.excel_as_df.columns.isin([colname]).any():
            if isinstance(value, str):
                dtype = 'str'
            elif isinstance(value, float):
                dtype = 'float'
            else:
                raise ValueError('Do not yet know how to post a value of type %s ' % str(type(value)))
            self.add_result_columns([colname], datatype=dtype)
        index = self.excel_as_df[self.excel_as_df[dataname] == dataid].index[0]
        self.excel_as_df.at[index, colname] = value
        
    def update_xlsx(self, filename, sheet):
        """
        Update the excel file by writing he current pandas dataframe
        to as an excel file. This is almost noe really needed
        
        Parameters
        ----------
        filename : str
            Full pant and extension for the excel file
        sheet : str or int (no default)
            Sheet to write the result to.
        
        """
        self.excel_as_df.to_excel(filename, sheet_name=sheet, index=False)
        
    # broken:
    # def read_ods(self, ods_filename):
    #     """
    #     Read an Open Document spreadsheet
    #     Assume that the first row contains the column headings for the data in the sheet
    #     Reads every row, saving a subset of the information to a dictionary
    #     The dictionary is keyed by the Day/Slice/Cell, and contains a dictionary with
    #     the following elements:
    #
    #     Parameter
    #     ---------
    #         ods_filename: the name of the file to read. The parent directory is set in the source
    #
    #     Return
    #     ------
    #         Dictionary: A dictionary with the spreadsheet information
    #             The structure is:
    #                 top level keys: sheet names
    #                     next level keys: day/slice/cell
    #                         containing a dict of data (genotype, description)
    #     """
    #     fn = os.path.join(basedir, ods_filename)
    #     doc = odf.opendoc(fn)
    #     result = {}
    #     # print("Spreadsheet %s contains %d sheets.\n" % (ods_filename, len(doc.sheets)))
    #     for sheet in doc.sheets:
    #         # print("-"*40)
    #         # print("Sheet name: '%s'" % sheet.name)
    #         # print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()) )
    #
    #         rt = sheet.row(0)
    #         titles = {t.value: int(i) for i, t in enumerate(rt)}  # titles of columns
    #         #  Assemble dictionary with key = day/slice/cell, containing dict{genotype: type, description: des}
    #         ss = {}
    #         for i, r in enumerate(sheet.rows()):
    #             if i == 0:
    #                 continue
    #             dayn = r[titles['Day']].value
    #             slicen = r[titles['Slice']].value
    #             celln = r[titles['Cell']].value
    #             genotype = r[titles['Genotype']].value
    #             description = r[titles['Day Description']].value
    #             vgenotype = r[titles['Verified Genotype']].value
    # #            species = r[titles['Species']].value
    #             if dayn is None or slicen is None or celln is None:  # probably just information for a human
    #                 continue
    #             thiskey = os.path.join(dayn.rstrip(), slicen.rstrip(), celln.rstrip())
    #             ss[thiskey] = {'genotype': genotype, 'description': description, 'verified': vgenotype}
    #         result[sheet] = ss
    #     return result
    #     #print ('Found {:d} Cells'.format(len(ss)))


if __name__ == '__main__':
    # D  = DataPlan(os.path.join('ephysanalysis', 'test_data', 'CS_CHL1_minis.py'))
    # D.make_xls(D.datasets)
    #
    # test routine -not for production. Use code like this in your own analysis script.
    D  = DataPlan('dataplan1.xlsx')
    D.post_result('CellID', 9, 'Rin', 52.3)
    #print( D.datasets)
    D.update_xlsx('dataplan2.xlsx', 'Dataplan')
    
    

