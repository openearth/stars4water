# -*- coding: utf-8 -*-
# Copyright notice
#   --------------------------------------------------------------------
#   Copyright (C) 2024 Deltares
#       Gerrit Hendriksen (gerrit.hendriksen@deltares.nl)
#       Ioanna Micha (Ioanna Micha)
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
import os
import requests
from owslib.csw import CatalogueServiceWeb
from owslib.fes import PropertyIsEqualTo, PropertyIsLike, BBox
from owslib.util import ServiceException

url = 'http://stars4water.openearth.nl/geonetwork/srv/eng/csw?'
url = 'http://localhost:8080/geonetwork/srv/eng/csw'
csw = CatalogueServiceWeb(url, version='2.0.2', timeout=60)

basedir = r'C:\projectinfo\eu\geodcat'
baseurl = 'http://localhost:8080/geonetwork/srv/api/records/'
psfix = '/formatters/eu-geodcat-ap'

csw.identification.type
[op.name for op in csw.operations]

try:
    result = csw.getrecords2(maxrecords=500, esn='full')
    csw.results
    for record in csw.records:
        print('exporting',csw.records[record].title)
        anid = csw.records[record].identifier
        xml_url = ''.join([baseurl,anid,psfix])
        try:
            # Send a GET request to retrieve the XML
            response = requests.get(xml_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Save the XML to a file
                filename = os.path.join(basedir, f"{anid}.xml")
                with open(filename, 'w') as f:
                    f.write(response.text)
                print(f"Exported {filename}")
            else:
                print(f"Failed to retrieve {xml_url} (status code {response.status_code})")
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving {xml_url}: {e}")
except TimeoutError as e:
    print(f"CSW server timed out {e}")
except ServiceException as e:
    print(f"CSW server error {e}")