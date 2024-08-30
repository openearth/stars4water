#  Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2024 Deltares Stars4Water
#   Gerrit.Hendriksen@deltares.nl
#
#   This library is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This library is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this library.  If not, see <http://www.gnu.org/licenses/>.
#   --------------------------------------------------------------------
#
# This tool is part of <a href="http://www.OpenEarth.eu">OpenEarthTools</a>.
# OpenEarthTools is an online collaboration to share and manage data and
# programming tools in an open source, version controlled environment.
# Sign up to recieve regular updates of this function, and to contribute
# your own tools.

"""
Convert metadata stored in Excel to xmls
"""
import os
import numpy as np
from datetime import time
import pandas as pd
import uuid

"""
directory where metadata is stored
"""
tdir = r"C:\projectinfo\eu\metadata"

"""setup mapping between metadata entries and xlsx"""
maptable = r".\mappingtable.xlsx"
d = (
    pd.read_excel(maptable, sheet_name="Sheet1", index_col=0, header=None)
    .transpose()
    .to_dict("records")[0]
)

"""
read the xslx with the metadata
Check column headers first, in some cases there are extra spaces there!!!
"""
mxlsx = r"C:\projectinfo\eu\stars4water\work\Datasets_for_portal20240726.xlsx"
dfm = pd.read_excel(
    mxlsx, skiprows=2, sheet_name="Sheet1", na_filter=False, index_col=0
)

"""
Read the xmltemplate
"""
txml = r".\iso19139_template.xml"

for r in range(1, len(dfm)):
    title = dfm["Title of the source"][r]
    if title != "":
        "create new file"
        fn = os.path.join(tdir, ".".join([title.replace(" ", ""), "xml"]))

        "open the template"
        fxml = open(txml, "r+")
        xml = fxml.read()
        fxml.close()

        "define unique identifier"
        uid = str(uuid.uuid4())
        print(uid)

        xml = xml.replace("{uuid}", uid)
        for k in d.keys():
            print("k", k)
            try:
                print("--".join([k, d[k], str(dfm[k][r])]))
                xml = xml.replace("{" + d[k] + "}", str(dfm[k][r]))
            except:
                print("it went wrong with", k, d[k], dfm[k][r])
        with open(fn, "w+", encoding="utf-8") as fm:
            fm.write(xml)
